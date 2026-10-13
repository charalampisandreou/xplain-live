import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from xplain.explainer import ADDITIVITY_TOLERANCE, ShapExplainer
from xplain.features import FEATURE_ORDER
from xplain.generator import generate_match
from xplain.scoring import XPassScorer
from xplain.schema import PassEvent
from xplain.train import (
    SPECS,
    TrainSpec,
    calibration_slope,
    evaluate,
    group_folds,
    group_holdout,
    pass_table,
    reliability_table,
    save,
    train,
)

SPEC = SPECS["pass"]


def tiny_table(n_matches: int = 10, per_match: int = 40, seed: int = 3) -> pd.DataFrame:
    """A few hundred rows with a learnable label, built from real generated passes of a few matches."""
    rng = np.random.default_rng(seed)
    events = generate_match(seed)[0]
    base = pass_table(events).head(n_matches * per_match).reset_index(drop=True)
    base["match_id"] = [f"m{i // per_match}" for i in range(len(base))]
    logit = 2.5 - 0.02 * base["start_x"] - 1.2 * base["under_pressure"] - 1.0 * base["height_High Pass"]
    base["completed"] = (rng.uniform(size=len(base)) < 1 / (1 + np.exp(-logit))).astype(int)
    return base


@pytest.fixture(scope="module")
def trained():
    df = tiny_table()
    return df, train(df, SPEC, depths=(2, 3), max_trees=60, early_stopping_rounds=10)


def test_pass_table_has_features_label_and_match():
    events = generate_match(5)[0]
    table = pass_table(events)
    assert list(table.columns) == [*FEATURE_ORDER, "completed", "match_id"]
    assert len(table) == sum(isinstance(e, PassEvent) for e in events)
    assert set(table["completed"]) <= {0, 1}


def test_holdout_split_has_no_match_on_both_sides():
    groups = np.repeat([f"m{i}" for i in range(12)], 30)
    fit_idx, val_idx = group_holdout(groups)
    assert set(groups[fit_idx]).isdisjoint(groups[val_idx])
    assert len(fit_idx) + len(val_idx) == len(groups)


def test_calibration_folds_have_no_match_on_both_sides():
    groups = np.repeat([f"m{i}" for i in range(10)], 25)
    folds = group_folds(groups, 5)
    assert len(folds) == 5
    for fit_idx, cal_idx in folds:
        assert set(groups[fit_idx]).isdisjoint(groups[cal_idx])
    assert sorted(np.concatenate([cal for _, cal in folds])) == list(range(len(groups)))


def test_model_has_the_structure_the_scorer_needs(trained):
    _, result = trained
    model = result.model
    assert isinstance(model, CalibratedClassifierCV)
    assert model.method == "isotonic"
    assert len(model.calibrated_classifiers_) == 5
    assert all(isinstance(c.estimator, XGBClassifier) for c in model.calibrated_classifiers_)
    assert result.max_depth in (2, 3) and 1 <= result.n_estimators <= 60


def test_saved_model_loads_in_the_scorer_and_explains_additively(trained, tmp_path):
    df, result = trained
    path = save(result, SPEC, tmp_path)
    assert json.loads((tmp_path / "feature_list.json").read_text(encoding="utf-8")) == FEATURE_ORDER
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training"]["max_depth"] == result.max_depth

    scorer = XPassScorer(path, tmp_path / "feature_list.json")
    events = generate_match(3)[0]
    passes = [e for e in events if isinstance(e, PassEvent)][:50]
    probs = [r.expected_probability for r in scorer.score([p.inputs for p in passes])]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert len(scorer.boosters) == 5
    contribs, margins = ShapExplainer(scorer.boosters).contributions([p.inputs for p in passes])
    assert np.max(np.abs(contribs.sum(axis=1) - margins)) < ADDITIVITY_TOLERANCE


def test_metrics_are_finite(trained):
    df, result = trained
    p = result.model.predict_proba(df[list(SPEC.features)])[:, 1]
    metrics = evaluate(df["completed"].to_numpy(), p)
    assert all(math.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["brier_constant_rate"] == pytest.approx(metrics["completion_rate"] * (1 - metrics["completion_rate"]))
    assert metrics["brier_skill"] == pytest.approx(1 - metrics["brier"] / metrics["brier_constant_rate"])
    table = reliability_table(df["completed"].to_numpy(), p)
    assert table["n"].sum() == len(df)


def test_calibration_slope_recovers_known_values():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.5, size=20000)
    p = 1 / (1 + np.exp(-x))
    y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-(0.3 + 0.6 * x)))).astype(float)
    a, b = calibration_slope(y, p)
    assert a == pytest.approx(0.3, abs=0.06)
    assert b == pytest.approx(0.6, abs=0.04)


def test_spec_is_generic():
    spec = TrainSpec(event_type="shot", features=("dist_to_goal", "angle_to_goal"), label="goal")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"dist_to_goal": rng.uniform(5, 40, 300), "angle_to_goal": rng.uniform(-1, 1, 300),
                       "match_id": np.repeat([f"m{i}" for i in range(10)], 30)})
    df["goal"] = (rng.uniform(size=300) < 1 / (1 + np.exp(0.15 * df["dist_to_goal"] - 1))).astype(int)
    result = train(df, spec, depths=(2,), max_trees=30, early_stopping_rounds=5)
    assert result.model.predict_proba(df[list(spec.features)]).shape == (300, 2)
