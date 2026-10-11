"""
Pass scoring with the calibrated xPass model.

xP is situational pass difficulty: the probability that a pass from this situation is
completed, estimated from start-of-pass information only (``PassInputs``). The model does not
know the intended target, so xP says nothing about whether a pass was a good choice.

``Scorer`` is the pluggable interface the API depends on; ``XPassScorer`` is the real one.
"""
from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from .features import FEATURE_ORDER, FeatureOrderError, build_features, validate_feature_order
from .schema import BodyPart, PassHeight, PassInputs, PlayPattern

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "xpass"
MODEL_GLOB = "xpass_production_*.joblib"
_TIMESTAMP = re.compile(r"xpass_production_(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})\.joblib$")


class ModelLoadError(RuntimeError):
    """The model or its feature list is missing, unreadable or does not match the feature builder."""


@dataclass(frozen=True)
class ScoreResult:
    expected_probability: float
    drivers: list = field(default_factory=list)


class Scorer(Protocol):
    model_version: str

    def score(self, inputs: Sequence[PassInputs]) -> list[ScoreResult]: ...


class XPassScorer:
    """Loads the xPass model once and scores PassInputs."""

    def __init__(self, model_path: str | Path, feature_list_path: str | Path):
        self.model_path = Path(model_path)
        self.feature_list_path = Path(feature_list_path)
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self.model_path}")
        if not self.feature_list_path.is_file():
            raise ModelLoadError(
                f"Feature list not found: {self.feature_list_path} (needed for model {self.model_path})"
            )
        try:
            validate_feature_order(self.feature_list_path)
        except (FeatureOrderError, ValueError) as exc:
            raise ModelLoadError(
                f"Feature list {self.feature_list_path} does not match the feature builder "
                f"(model {self.model_path}): {exc}"
            ) from exc

        self._model = joblib.load(self.model_path)
        self.model_version = self.model_path.name
        self.n_features = len(FEATURE_ORDER)
        # warm-up: the first predict_proba call is noticeably slower than later ones
        self.score([PassInputs(start_x=60.0, start_y=40.0, height=PassHeight.GROUND, body_part=BodyPart.FOOT,
                               play_pattern=PlayPattern.REGULAR, under_pressure=False)])

    def score(self, inputs: Sequence[PassInputs]) -> list[ScoreResult]:
        if not inputs:
            return []
        features = build_features(inputs)  # accepts PassInputs only
        probs = np.asarray(self._model.predict_proba(features)[:, 1], dtype=float)
        if not np.all(np.isfinite(probs)) or np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError(f"Model {self.model_version} returned probabilities outside [0, 1]: {probs}")
        return [ScoreResult(expected_probability=float(p)) for p in probs]

    @property
    def boosters(self) -> tuple[xgb.Booster, ...]:
        """The XGBoost booster of each calibration fold, for explanations (SHAP). Read-only."""
        if not isinstance(self._model, CalibratedClassifierCV):
            raise ModelLoadError(
                f"Model {self.model_version} is a {type(self._model).__name__}, not a CalibratedClassifierCV"
            )
        estimators = [pair.estimator for pair in self._model.calibrated_classifiers_]
        if not estimators or not all(isinstance(e, xgb.XGBClassifier) for e in estimators):
            raise ModelLoadError(
                f"Model {self.model_version} is not calibrated over XGBClassifier fold models: "
                f"{[type(e).__name__ for e in estimators]}"
            )
        return tuple(e.get_booster() for e in estimators)


def _model_timestamp(path: Path) -> datetime:
    m = _TIMESTAMP.search(path.name)
    return datetime.strptime(m.group(1), "%d-%m-%Y_%H-%M-%S") if m else datetime.min


def find_default_model(models_dir: str | Path = MODELS_DIR) -> tuple[Path, Path]:
    """Return (model_path, feature_list_path).

    Uses XPLAIN_MODEL_PATH / XPLAIN_FEATURE_LIST when set; otherwise the newest
    xpass_production_<dd-mm-yyyy_hh-mm-ss>.joblib in models/xpass/ and the feature_list.json next to it.
    """
    env_model = os.environ.get("XPLAIN_MODEL_PATH")
    env_features = os.environ.get("XPLAIN_FEATURE_LIST")
    if env_model:
        model_path = Path(env_model)
    else:
        candidates = sorted(Path(models_dir).glob(MODEL_GLOB), key=lambda p: (_model_timestamp(p), p.name))
        if not candidates:
            raise ModelLoadError(f"No {MODEL_GLOB} found in {Path(models_dir)} (see models/README.md)")
        model_path = candidates[-1]
    feature_list_path = Path(env_features) if env_features else model_path.parent / "feature_list.json"
    return model_path, feature_list_path
