"""
Train and evaluate an event-outcome model (xPass now; xG later) on generated matches.

The model is the structure XPassScorer and the Explainer expect: a CalibratedClassifierCV
(isotonic) over XGBClassifier fold models, so scorer.boosters gives one booster per fold.

Training, for one TrainSpec (event type, feature list, label column, group column):
  1. Hold out a match-grouped validation split; fit XGBClassifier with early stopping for each
     depth in DEPTHS and keep the depth and tree count with the lowest validation log-loss.
  2. Refit on all training rows inside CalibratedClassifierCV with N_FOLDS GroupKFold folds by
     match, so a match is never in both the fit and the calibration part of a fold.

    python -m xplain.train --train data/xpass_train.csv --holdout data/xpass_holdout.csv \\
                           --out models/xpass_v2 --report reports/xpass_v2_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from .features import FEATURE_ORDER, build_features
from .schema import PassEvent

N_FOLDS = 5
VAL_FRACTION = 0.2
DEPTHS: tuple[int, ...] = (3, 4, 5)
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50
LEARNING_RATE = 0.05
SEED = 7
EPS = 1e-6          # probability clip for logits and log-loss
MODEL_FILE = "xpass_v2.joblib"


@dataclass(frozen=True)
class TrainSpec:
    """What to train: rows of one event type, these feature columns, this 0/1 label, grouped by match."""

    event_type: str
    features: tuple[str, ...]
    label: str
    group: str = "match_id"


SPECS: dict[str, TrainSpec] = {
    "pass": TrainSpec(event_type="pass", features=tuple(FEATURE_ORDER), label="completed"),
}


@dataclass
class TrainResult:
    model: CalibratedClassifierCV
    max_depth: int
    n_estimators: int
    val_logloss_by_depth: dict[int, float]
    n_rows: int
    n_groups: int
    train_seconds: float
    params: dict = field(default_factory=dict)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("model")
        return d


# --- data -------------------------------------------------------------------------------------

def pass_table(events: Sequence) -> pd.DataFrame:
    """One row per pass: the model features (from PassInputs only), the label and the match id."""
    passes = [e for e in events if isinstance(e, PassEvent)]
    table = build_features([p.inputs for p in passes])
    table["completed"] = [int(p.details.completed) for p in passes]
    table["match_id"] = [p.match_id for p in passes]
    return table


# --- splits -----------------------------------------------------------------------------------

def group_holdout(groups: Sequence, fraction: float = VAL_FRACTION, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """(fit, validation) row indices with every group on one side only."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    fit_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return fit_idx, val_idx


def group_folds(groups: Sequence, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold (fit, calibrate) splits by group, as a list CalibratedClassifierCV accepts as cv."""
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups))


# --- training ---------------------------------------------------------------------------------

def _xgb(**kw) -> XGBClassifier:
    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", learning_rate=LEARNING_RATE,
                         subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=SEED, **kw)


def train(df: pd.DataFrame, spec: TrainSpec, *, depths: Sequence[int] = DEPTHS, max_trees: int = MAX_TREES,
          early_stopping_rounds: int = EARLY_STOPPING_ROUNDS, n_folds: int = N_FOLDS) -> TrainResult:
    """Choose depth and tree count by early stopping on a match-grouped split, then fit the calibrated model."""
    start = time.perf_counter()
    X, y, groups = df[list(spec.features)], df[spec.label].to_numpy(), df[spec.group].to_numpy()
    fit_idx, val_idx = group_holdout(groups)

    best: tuple[float, int, int] | None = None   # (val log-loss, depth, trees)
    by_depth: dict[int, float] = {}
    for depth in depths:
        clf = _xgb(n_estimators=max_trees, max_depth=depth, early_stopping_rounds=early_stopping_rounds)
        clf.fit(X.iloc[fit_idx], y[fit_idx], eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        score = float(clf.best_score)
        by_depth[depth] = score
        if best is None or score < best[0]:
            best = (score, depth, int(clf.best_iteration) + 1)
    _, depth, trees = best

    model = CalibratedClassifierCV(_xgb(n_estimators=trees, max_depth=depth), method="isotonic",
                                   cv=group_folds(groups, n_folds))
    model.fit(X, y)
    return TrainResult(model=model, max_depth=depth, n_estimators=trees, val_logloss_by_depth=by_depth,
                       n_rows=len(df), n_groups=len(set(groups)), train_seconds=time.perf_counter() - start,
                       params={"learning_rate": LEARNING_RATE, "subsample": 0.9, "colsample_bytree": 0.9,
                               "early_stopping_rounds": early_stopping_rounds, "n_folds": n_folds,
                               "calibration": "isotonic", "seed": SEED})


def save(result: TrainResult, spec: TrainSpec, out_dir: Path, model_file: str = MODEL_FILE) -> Path:
    """Write the model, feature_list.json and metrics.json; return the model path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / model_file
    joblib.dump(result.model, path)
    (out_dir / "feature_list.json").write_text(json.dumps(list(spec.features), indent=2) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"spec": asdict(spec), "training": result.summary()},
                                                     indent=2) + "\n", encoding="utf-8")
    return path


# --- evaluation -------------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """(a, b) of the logistic fit logit(P(y=1)) = a + b * logit(p), by maximum likelihood."""
    x = _logit(p)

    def nll(theta: np.ndarray) -> float:
        z = theta[0] + theta[1] * x
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    def grad(theta: np.ndarray) -> np.ndarray:
        r = 1.0 / (1.0 + np.exp(-(theta[0] + theta[1] * x))) - y
        return np.array([r.sum(), (r * x).sum()])

    fit = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    return float(fit.x[0]), float(fit.x[1])


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Discrimination and calibration on held-out rows, with the constant-rate baseline."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    ybar = float(y.mean())
    brier = float(brier_score_loss(y, p))
    brier0 = ybar * (1 - ybar)
    a, b = calibration_slope(y, p)
    return {
        "n": int(len(y)),
        "completion_rate": ybar,
        "mean_prediction": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "log_loss_constant_rate": -(ybar * math.log(ybar) + (1 - ybar) * math.log(1 - ybar)),
        "brier": brier,
        "brier_constant_rate": brier0,
        "brier_skill": 1 - brier / brier0,
        "calibration_intercept": a,
        "calibration_slope": b,
    }


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per equal-width bin of p: count, mean prediction, observed rate."""
    bins = np.clip((np.asarray(p) * n_bins).astype(int), 0, n_bins - 1)
    df = pd.DataFrame({"bin": bins, "p": p, "y": y})
    return df.groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean")).reset_index()


def plot_reliability(table: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(table["mean_p"], table["observed"], marker="o", label="model")
    for _, row in table.iterrows():
        ax.annotate(f"{int(row['n'])}", (row["mean_p"], row["observed"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    ax.set_xlabel("predicted completion probability (bin mean)")
    ax.set_ylabel("observed completion rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- CLI --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a calibrated XGBoost outcome model on generated matches.")
    parser.add_argument("--event", default="pass", choices=sorted(SPECS))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="model directory")
    parser.add_argument("--report", type=Path, required=True, help="hold-out metrics JSON")
    args = parser.parse_args(argv)
    spec = SPECS[args.event]

    train_df = pd.read_csv(args.train)
    holdout_df = pd.read_csv(args.holdout)
    overlap = set(train_df[spec.group]) & set(holdout_df[spec.group])
    if overlap:
        raise SystemExit(f"train and hold-out share matches: {sorted(overlap)[:5]}")

    result = train(train_df, spec)
    print(f"trained on {result.n_rows} rows / {result.n_groups} matches in {result.train_seconds:.1f} s: "
          f"max_depth {result.max_depth}, n_estimators {result.n_estimators}, "
          f"val log-loss by depth {result.val_logloss_by_depth}")
    path = save(result, spec, args.out)
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB)")

    p = result.model.predict_proba(holdout_df[list(spec.features)])[:, 1]
    y = holdout_df[spec.label].to_numpy()
    metrics = evaluate(y, p)
    table = reliability_table(y, p)
    png = args.report.with_suffix(".png")
    plot_reliability(table, png, f"{args.out.name}: hold-out reliability ({len(y)} {spec.event_type}es)")
    report = {"model": str(path.as_posix()), "model_bytes": path.stat().st_size,
              "holdout_matches": int(holdout_df[spec.group].nunique()), "holdout": metrics,
              "reliability": table.to_dict(orient="records"), "training": result.summary()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.report} and {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
