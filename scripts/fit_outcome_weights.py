"""
Fit the synthetic generator's pass-completion weights on StatsBomb open data.

Reads StatsBomb event files one match at a time, keeps only Pass events and fits an
unpenalised logistic regression with the same terms as the generator's hidden mechanism
(see src/xplain/config.py), minus the latent pass_skill:

    logit = intercept + w_length*length + w_forward*max(end_x - start_x, 0)
            + w_pressure*under_pressure + w_high*is_high + w_low*is_low + restart bonuses

Five variants are fitted: A (all terms), B (start-only: no length, no forward_gain),
C (A + start_x, dist_to_goal, angle_to_goal, defined as in src/xplain/features.py), C2 (C's
terms with length, forward_gain, under_pressure, is_high and is_low constrained to <= 0, fitted
with scipy's L-BFGS-B on the same log-loss) and D (start-only information, as the xPass model
sees a pass: B + start_x, dist_to_goal, angle_to_goal, with under_pressure, is_high and is_low
constrained to <= 0). The real completion rate per bucket (BUCKETS) is
saved too. Results go to reports/outcome_fit.json.
Nothing is written back to config.py.

Pass selection and the completion label copy xPass-Project's ETL
(src/data_analysis_pipeline.py, process_match_json):
    - event type name == "Pass"
    - drop passes without a start location or without pass.end_location
    - completed = 1 when pass.outcome is absent, else 0 (Incomplete, Out, Pass Offside,
      Unknown and Injury Clearance all count as failed)

Usage:
    python scripts/fit_outcome_weights.py --data-dir <path to statsbomb_free>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# StatsBomb pass.type names (verified against the raw JSON: note "Throw-in", not "Throw In")
RESTART_TYPES: dict[str, str] = {
    "kick_off": "Kick Off",
    "throw_in": "Throw-in",
    "goal_kick": "Goal Kick",
    "free_kick": "Free Kick",
    "corner": "Corner",
}
RESTARTS: tuple[str, ...] = (*RESTART_TYPES, "keeper")

FEATURES_A: tuple[str, ...] = ("length", "forward_gain", "under_pressure", "is_high", "is_low", *RESTARTS)
FEATURES_B: tuple[str, ...] = ("under_pressure", "is_high", "is_low", *RESTARTS)
FEATURES_C: tuple[str, ...] = (*FEATURES_A, "start_x", "dist_to_goal", "angle_to_goal")
# C2: these terms may only make a pass harder (coefficient <= 0); the rest are free
NONPOSITIVE_C2: tuple[str, ...] = ("length", "forward_gain", "under_pressure", "is_high", "is_low")
# D: only what is known when the pass is played (no length, no forward_gain)
FEATURES_D: tuple[str, ...] = (*FEATURES_B, "start_x", "dist_to_goal", "angle_to_goal")
NONPOSITIVE_D: tuple[str, ...] = ("under_pressure", "is_high", "is_low")
FINAL_THIRD_X = 80.0
# written to the report instead of --data-dir, so no local path ends up in the repository
DATA_DIR_LABEL = "statsbomb_free (StatsBomb open data, local copy)"

# Goal centre, as in src/xplain/features.py
GOAL_X, GOAL_Y = 120.0, 40.0

# Completion-rate buckets, shared with scripts/compare_variants.py
HEIGHT_BUCKETS: tuple[str, ...] = ("ground", "low", "high")
RESTART_BUCKETS: tuple[str, ...] = (*RESTARTS, "none")
LENGTH_EDGES: tuple[float, ...] = (10.0, 20.0, 30.0, 45.0)
LENGTH_BUCKETS: tuple[str, ...] = ("0-10", "10-20", "20-30", "30-45", "45+")
START_X_EDGES: tuple[float, ...] = (30.0, 60.0, 90.0)
START_X_BUCKETS: tuple[str, ...] = ("0-30", "30-60", "60-90", "90-120")
PRESSURE_BUCKETS: tuple[str, ...] = ("0", "1")
BUCKETS: dict[str, tuple[str, ...]] = {
    "height": HEIGHT_BUCKETS,
    "restart": RESTART_BUCKETS,
    "length": LENGTH_BUCKETS,
    "start_x": START_X_BUCKETS,
    "under_pressure": PRESSURE_BUCKETS,
}

# Hand-set values from GenConfig, keyed by feature name
CURRENT: dict[str, float] = {
    "intercept": 2.9,
    "length": -0.045,
    "forward_gain": -0.03,
    "under_pressure": -0.55,
    "is_high": -0.35,
    "is_low": -0.25,
    "kick_off": 1.5,
    "throw_in": 0.3,
    "goal_kick": 0.0,
    "free_kick": 0.5,
    "corner": -0.5,
    "keeper": 0.5,
    "start_x": 0.0,
    "dist_to_goal": 0.0,
    "angle_to_goal": 0.0,
}


def _name(obj: Any) -> str | None:
    """StatsBomb nests categories as {"id": .., "name": ..}; return the name or None."""
    return obj.get("name") if isinstance(obj, dict) else None


def pass_features(event: dict[str, Any], is_keeper: bool = False) -> dict[str, float] | None:
    """One StatsBomb Pass event -> feature row, or None when xPass's ETL would drop it."""
    p = event.get("pass")
    start, end = event.get("location"), p.get("end_location") if isinstance(p, dict) else None
    if not isinstance(start, list) or not isinstance(end, list) or len(start) < 2 or len(end) < 2:
        return None
    sx, sy, ex, ey = float(start[0]), float(start[1]), float(end[0]), float(end[1])
    height = _name(p.get("height"))
    ptype = _name(p.get("type"))
    row = {
        "completed": float(p.get("outcome") is None),
        "length": math.hypot(ex - sx, ey - sy),
        "forward_gain": max(ex - sx, 0.0),
        "under_pressure": float(bool(event.get("under_pressure", False))),
        "is_high": float(height == "High Pass"),
        "is_low": float(height == "Low Pass"),
    }
    for key, type_name in RESTART_TYPES.items():
        row[key] = float(ptype == type_name)
    row["keeper"] = float(is_keeper)
    row["start_x"] = sx
    row["dist_to_goal"] = math.hypot(GOAL_X - sx, GOAL_Y - sy)
    row["angle_to_goal"] = math.atan2(GOAL_Y - sy, GOAL_X - sx)
    return row


def assign_buckets(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bucket label per pass for each dimension in BUCKETS.

    cols needs is_high, is_low, length, start_x, under_pressure and one 0/1 array per name in
    RESTARTS (at most one of them set per pass).
    """
    height = np.full(len(cols["length"]), "ground", dtype=object)
    height[cols["is_low"] == 1] = "low"
    height[cols["is_high"] == 1] = "high"
    restart = np.full(len(cols["length"]), "none", dtype=object)
    for r in RESTARTS:
        restart[cols[r] == 1] = r
    return {
        "height": height,
        "restart": restart,
        "length": np.asarray(LENGTH_BUCKETS, dtype=object)[np.digitize(cols["length"], LENGTH_EDGES)],
        "start_x": np.asarray(START_X_BUCKETS, dtype=object)[np.digitize(cols["start_x"], START_X_EDGES)],
        "under_pressure": np.where(cols["under_pressure"] == 1, "1", "0").astype(object),
    }


def bucket_rates(labels: dict[str, np.ndarray], values: np.ndarray) -> dict[str, dict[str, dict[str, Any]]]:
    """{dimension: {bucket: {"n": passes, "rate": mean of values or None}}} in BUCKETS order."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim, names in BUCKETS.items():
        out[dim] = {}
        for name in names:
            mask = labels[dim] == name
            n = int(mask.sum())
            out[dim][name] = {"n": n, "rate": float(values[mask].mean()) if n else None}
    return out


def match_rows(events: Iterable[dict[str, Any]]) -> Iterator[dict[str, float]]:
    """Feature rows for every kept pass in one match.

    Keeper flag: the first Pass of a possession whose play pattern is "From Keeper", made by
    the possession team's Goalkeeper, with no pass.type (so it is not also a set piece).
    """
    seen_possessions: set[Any] = set()
    for e in events:
        if _name(e.get("type")) != "Pass":
            continue
        poss = e.get("possession")
        first = poss not in seen_possessions
        seen_possessions.add(poss)
        is_keeper = (
            first
            and _name(e.get("play_pattern")) == "From Keeper"
            and _name(e.get("position")) == "Goalkeeper"
            and _name((e.get("pass") or {}).get("type")) is None
            and (e.get("team") or {}).get("id") == (e.get("possession_team") or {}).get("id")
        )
        row = pass_features(e, is_keeper)
        if row is not None:
            yield row


def fit_logit(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Unpenalised logistic regression; returns (intercept, coefs) on the original feature scale.

    scikit-learn 1.9 deprecates penalty=None; C=np.inf is the documented replacement.
    Features are standardised for the solver and the coefficients mapped back afterwards.
    """
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, tol=1e-8)
    model.fit((X - mean) / sd, y)
    coefs = model.coef_[0] / sd
    intercept = float(model.intercept_[0] - np.sum(coefs * mean))
    return intercept, coefs


def fit_logit_bounded(X: np.ndarray, y: np.ndarray, nonpositive: Iterable[bool]) -> tuple[float, np.ndarray]:
    """Unpenalised logistic regression with coefficient j bounded to <= 0 where nonpositive[j].

    Minimises the mean log-loss with L-BFGS-B on standardised features (dividing by a positive
    sd keeps each coefficient's sign, so the bounds carry over); coefficients are mapped back
    to the original scale. Same return shape as fit_logit.
    """
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (X - mean) / sd
    n = len(y)

    def loss_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = w[0] + Xs @ w[1:]
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        r = (expit(z) - y) / n
        return loss, np.concatenate(([r.sum()], Xs.T @ r))

    bounds = [(None, None)] + [(None, 0.0) if neg else (None, None) for neg in nonpositive]
    if len(bounds) != X.shape[1] + 1:
        raise ValueError("nonpositive must have one entry per column of X")
    res = minimize(loss_grad, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9})
    if not res.success:
        print(f"warning: L-BFGS-B did not converge: {res.message}", file=sys.stderr)
    coefs = res.x[1:] / sd
    intercept = float(res.x[0] - np.sum(coefs * mean))
    return intercept, coefs


def predict_proba(intercept: float, coefs: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(intercept + X @ coefs)))


@dataclass
class Corpus:
    columns: tuple[str, ...]
    chunks: list[np.ndarray] = field(default_factory=list)
    matches: int = 0
    skipped_files: list[str] = field(default_factory=list)

    def add_match(self, rows: Iterable[dict[str, float]]) -> None:
        data = [[r[c] for c in self.columns] for r in rows]
        if data:
            self.chunks.append(np.asarray(data, dtype=np.float32))
        self.matches += 1

    def array(self) -> np.ndarray:
        return np.concatenate(self.chunks).astype(np.float64)


def load_corpus(data_dir: Path, limit: int | None) -> Corpus:
    files = sorted(data_dir.rglob("*.json"))
    if limit:
        files = files[:limit]
    corpus = Corpus(columns=("completed", *FEATURES_C))
    t0 = time.time()
    for i, path in enumerate(files, 1):
        try:
            with path.open(encoding="utf-8") as f:
                events = json.load(f)
        except (OSError, json.JSONDecodeError):
            corpus.skipped_files.append(str(path))
            continue
        corpus.add_match(match_rows(events))
        del events
        if i % 250 == 0 or i == len(files):
            print(f"  read {i}/{len(files)} files ({time.time() - t0:.0f}s)", file=sys.stderr)
    return corpus


def fit_variant(data: np.ndarray, columns: tuple[str, ...], features: tuple[str, ...],
                nonpositive: tuple[str, ...] = ()) -> dict[str, Any]:
    idx = [columns.index(f) for f in features]
    X, y = data[:, idx], data[:, columns.index("completed")]
    if nonpositive:
        intercept, coefs = fit_logit_bounded(X, y, [f in nonpositive for f in features])
    else:
        intercept, coefs = fit_logit(X, y)
    p = np.clip(predict_proba(intercept, coefs, X), 1e-12, 1 - 1e-12)
    out = {
        "features": list(features),
        "coefficients": {"intercept": intercept, **{f: float(c) for f, c in zip(features, coefs)}},
        "mean_predicted": float(p.mean()),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }
    if nonpositive:
        out["constrained_nonpositive"] = list(nonpositive)
        out["active_bounds"] = [f for f, c in zip(features, coefs) if f in nonpositive and abs(c) < 1e-9]
    return out


def print_table(fits: dict[str, dict[str, Any]]) -> None:
    fmt = lambda v: f"{v:>10.4f}" if v is not None else f"{'-':>10}"
    print(f"\n{'term':<16}{'hand-set':>10}" + "".join(f"{name:>10}" for name in fits))
    print("-" * (26 + 10 * len(fits)))
    for r in ("intercept", *FEATURES_C):
        print(f"{r:<16}{CURRENT[r]:>10.3f}" + "".join(fmt(f["coefficients"].get(r)) for f in fits.values()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1], formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="StatsBomb event files (searched recursively)")
    parser.add_argument("--out", type=Path, default=Path("reports/outcome_fit.json"))
    parser.add_argument("--limit", type=int, default=None, help="only read the first N files (for quick checks)")
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        parser.error(f"--data-dir not found: {args.data_dir}")

    corpus = load_corpus(args.data_dir, args.limit)
    data, cols = corpus.array(), corpus.columns
    col = {c: data[:, i] for i, c in enumerate(cols)}
    y, length = col["completed"], col["length"]
    counts = {r: int(col[r].sum()) for r in ("under_pressure", "is_high", "is_low", *RESTARTS)}
    restart_rates = {r: float(y[col[r] == 1].mean()) if counts[r] else None for r in RESTARTS}

    fit_a = fit_variant(data, cols, FEATURES_A)
    fit_b = fit_variant(data, cols, FEATURES_B)
    fit_c = fit_variant(data, cols, FEATURES_C)
    fit_c2 = fit_variant(data, cols, FEATURES_C, nonpositive=NONPOSITIVE_C2)
    fit_d = fit_variant(data, cols, FEATURES_D, nonpositive=NONPOSITIVE_D)

    summary = {
        "matches": corpus.matches,
        "skipped_files": corpus.skipped_files,
        "passes": int(len(y)),
        "completion_rate": float(y.mean()),
        "mean_length": float(length.mean()),
        "mean_length_completed": float(length[y == 1].mean()),
        "mean_length_failed": float(length[y == 0].mean()),
        "share_high": float(col["is_high"].mean()),
        "share_low": float(col["is_low"].mean()),
        "share_under_pressure": float(col["under_pressure"].mean()),
        "share_final_third": float((col["start_x"] >= FINAL_THIRD_X).mean()),
        "passes_per_match": float(len(y) / corpus.matches) if corpus.matches else None,
        "feature_counts": counts,
        "restart_completion_rates": restart_rates,
    }
    report = {
        "source": "StatsBomb open data",
        "data_dir": DATA_DIR_LABEL,  # never the local path given on the command line
        "units": "StatsBomb pitch units (120 x 80), same frame as the generator",
        "regularisation": "none (LogisticRegression C=np.inf, scikit-learn 1.9.0)",
        "summary": summary,
        "real_buckets": bucket_rates(assign_buckets(col), y),
        "hand_set": CURRENT,
        "variant_A_all": fit_a,
        "variant_B_start_only": fit_b,
        "variant_C_all_plus_location": fit_c,
        "variant_C2_sign_constrained": fit_c2,
        "variant_D_start_only_location": fit_d,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nmatches: {corpus.matches:,}  passes: {len(y):,}  skipped files: {len(corpus.skipped_files)}")
    print(f"completion rate: {summary['completion_rate']:.4f}")
    fits = {"A": fit_a, "B": fit_b, "C": fit_c, "C2": fit_c2, "D": fit_d}
    print("mean predicted:  " + "  ".join(f"{k} {f['mean_predicted']:.4f}" for k, f in fits.items()))
    print("log-loss:        " + "  ".join(f"{k} {f['log_loss']:.4f}" for k, f in fits.items()))
    print(f"C2 active bounds (coefficient pinned at 0): {fit_c2['active_bounds']}")
    print(f"D active bounds (coefficient pinned at 0): {fit_d['active_bounds']}")
    print(f"mean length: completed {summary['mean_length_completed']:.2f}  failed {summary['mean_length_failed']:.2f}")
    print("feature counts:", counts)
    print("restart completion rates:", {k: round(v, 4) if v is not None else None for k, v in restart_rates.items()})
    print_table({"A: all": fit_a, "B: start": fit_b, "C: A+loc": fit_c, "C2: signs": fit_c2,
                 "D: B+loc": fit_d})
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
