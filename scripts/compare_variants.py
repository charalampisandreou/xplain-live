"""
Compare fitted pass-completion weights inside the synthetic generator.

For the old hand-set weights of generator 0.1.0 ("H", stored as hand_set in the report) and
variants A, B, C, C2 and D from reports/outcome_fit.json, map the coefficients onto GenConfig,
generate matches for seeds 1..20, score every pass with the xPass model (XPassScorer, no HTTP)
and compare against the real StatsBomb buckets saved by scripts/fit_outcome_weights.py.
Nothing is written. Since generator 0.1.1 the GenConfig defaults are variant C.

Modes:
    (default)      summary per variant, MAD against the real buckets, calibration of xP on the
                   synthetic passes (reliability table, log-loss, Brier, calibration slope) for
                   the --calibrate variants (repeated as a summary line on --check-seeds), and
                   bucket tables for the --buckets variants
    --reliability  reliability table for the current GenConfig defaults only
    --diagnose     for the current GenConfig defaults: pass mix and mean logit contribution of
                   every term, for the passes with xP < 0.4 and those with xP > 0.8

Restart passes are not marked in the events file, so each match runs through a subclass of the
generator's _Match that logs, at every pass, the restart flag and each term of the completion
logit (checked against the logit the generator actually used). It makes no extra random draws,
so the passes are the same as generate_match's.

Usage:
    python scripts/compare_variants.py [--fit reports/outcome_fit.json] [--seeds 20]
    python scripts/compare_variants.py --reliability [--seeds 20]
    python scripts/compare_variants.py --diagnose [--seeds 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import xplain.generator as gen  # noqa: E402
from fit_outcome_weights import BUCKETS, FINAL_THIRD_X, RESTARTS, assign_buckets, bucket_rates  # noqa: E402
from xplain.config import GenConfig  # noqa: E402
from xplain.generator import _Match  # noqa: E402
from xplain.schema import PITCH_LENGTH, PITCH_WIDTH, BodyPart, PassEvent, PassHeight, PlayPattern  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402

VARIANT_KEYS: dict[str, str] = {
    "A": "variant_A_all",
    "B": "variant_B_start_only",
    "C": "variant_C_all_plus_location",
    "C2": "variant_C2_sign_constrained",
    "D": "variant_D_start_only_location",
}

# fitted coefficient name -> GenConfig field (w_skill is kept at its current value)
COEF_TO_FIELD: dict[str, str] = {
    "intercept": "intercept",
    "length": "w_length",
    "forward_gain": "w_forward",
    "under_pressure": "w_pressure",
    "is_high": "w_high",
    "is_low": "w_low",
    "kick_off": "bonus_kick_off",
    "throw_in": "bonus_throw_in",
    "goal_kick": "bonus_goal_kick",
    "free_kick": "bonus_free_kick",
    "corner": "bonus_corner",
    "keeper": "bonus_keeper",
    "start_x": "w_start_x",
    "dist_to_goal": "w_dist_goal",
    "angle_to_goal": "w_angle_goal",
}

PATTERN_TO_RESTART: dict[PlayPattern, str] = {
    PlayPattern.KICK_OFF: "kick_off",
    PlayPattern.THROW_IN: "throw_in",
    PlayPattern.GOAL_KICK: "goal_kick",
    PlayPattern.FREE_KICK: "free_kick",
    PlayPattern.CORNER: "corner",
    PlayPattern.KEEPER: "keeper",
}

# terms of the generator's completion logit, in the order they are printed
TERMS: tuple[str, ...] = ("intercept", "length", "forward", "pressure", "high", "low", "restart",
                          "start_x", "dist_to_goal", "angle_to_goal", "skill")
GOAL_Y = PITCH_WIDTH / 2


def variant_config(coefs: dict[str, float]) -> GenConfig:
    """GenConfig with every mapped field taken from coefs; a term the variant lacks is set to 0.0."""
    return replace(GenConfig(), **{field: float(coefs.get(name, 0.0)) for name, field in COEF_TO_FIELD.items()})


class _LoggedMatch(_Match):
    """_Match that records, per pass, the restart flag and every term of the completion logit."""

    def __init__(self, seed: int, cfg: GenConfig):
        super().__init__(seed, cfg)
        self.restart_log: list[bool] = []
        self.term_log: list[dict[str, float]] = []
        self._design: tuple | None = None

    def _design_pass(self, position: str, sx: float, sy: float):
        self._design = super()._design_pass(position, sx, sy)
        return self._design

    def _pass(self) -> None:
        cfg = self.cfg
        sx, sy = self.x, self.y
        skill = self.teams[self.poss]["players"][self.holder]["pass_skill"]
        bonus = self._restart_bonus()
        self.restart_log.append(bool(self.restart))

        # capture the logit the generator feeds to _sigmoid for the completion draw
        seen: list[float] = []
        original = gen._sigmoid
        gen._sigmoid = lambda z: (seen.append(z), original(z))[1]
        try:
            super()._pass()
        finally:
            gen._sigmoid = original

        # rebuild the intended end point exactly as _Match._pass does (before any out-of-play move)
        length, direction, height, _ = self._design
        m = cfg.end_margin
        ex = gen._r2(gen._clip(sx + length * math.cos(direction), m, PITCH_LENGTH - m))
        ey = gen._r2(gen._clip(sy + length * math.sin(direction), m, PITCH_WIDTH - m))
        length = gen._r2(math.hypot(ex - sx, ey - sy))
        pressure = next(r[5]["inputs"].under_pressure for r in reversed(self.records) if r[4] is PassEvent)
        terms = {
            "intercept": cfg.intercept,
            "length": cfg.w_length * length,
            "forward": cfg.w_forward * max(ex - sx, 0.0),
            "pressure": cfg.w_pressure * pressure,
            "high": cfg.w_high * (height == PassHeight.HIGH),
            "low": cfg.w_low * (height == PassHeight.LOW),
            "restart": bonus,
            "start_x": cfg.w_start_x * sx,
            "dist_to_goal": cfg.w_dist_goal * math.hypot(PITCH_LENGTH - sx, GOAL_Y - sy),
            "angle_to_goal": cfg.w_angle_goal * math.atan2(GOAL_Y - sy, PITCH_LENGTH - sx),
            "skill": cfg.w_skill * skill,
        }
        if len(seen) != 1 or abs(sum(terms.values()) - seen[0]) > 1e-9:
            raise RuntimeError(f"logit terms do not reproduce the generator's logit: {seen} vs {terms}")
        self.term_log.append(terms)


@dataclass
class MatchPasses:
    cols: dict[str, np.ndarray]
    completed: np.ndarray
    xp: np.ndarray
    terms: dict[str, np.ndarray]


def simulate(seed: int, cfg: GenConfig, scorer: XPassScorer) -> MatchPasses:
    match = _LoggedMatch(seed, cfg)
    match.run()
    passes = [e for e in match.build_events() if isinstance(e, PassEvent)]
    if len(passes) != len(match.restart_log):
        raise RuntimeError(f"seed {seed}: {len(passes)} pass events but {len(match.restart_log)} logged passes")

    restart = [PATTERN_TO_RESTART.get(p.inputs.play_pattern) if r else None
               for p, r in zip(passes, match.restart_log)]
    cols = {
        "is_high": np.array([p.inputs.height == PassHeight.HIGH for p in passes], dtype=float),
        "is_low": np.array([p.inputs.height == PassHeight.LOW for p in passes], dtype=float),
        "length": np.array([p.details.length for p in passes], dtype=float),
        "forward_gain": np.array([max(p.details.end_x - p.inputs.start_x, 0.0) for p in passes], dtype=float),
        "start_x": np.array([p.inputs.start_x for p in passes], dtype=float),
        "under_pressure": np.array([p.inputs.under_pressure for p in passes], dtype=float),
        "is_head": np.array([p.inputs.body_part == BodyPart.HEAD for p in passes], dtype=float),
        **{name: np.array([r == name for r in restart], dtype=float) for name in RESTARTS},
    }
    completed = np.array([p.details.completed for p in passes], dtype=float)
    xp = np.array([s.expected_probability for s in scorer.score([p.inputs for p in passes])], dtype=float)
    terms = {t: np.array([row[t] for row in match.term_log], dtype=float) for t in TERMS}
    return MatchPasses(cols, completed, xp, terms)


def calibration(xp: np.ndarray, completed: np.ndarray) -> dict[str, float]:
    """Log-loss and Brier score of xP on these outcomes, and the calibration slope: the coefficient
    of an unpenalised logistic regression of the outcome on logit(xP), with intercept."""
    p = np.clip(xp, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, max_iter=1000).fit(z, completed)
    return {
        "log_loss": float(-np.mean(completed * np.log(p) + (1 - completed) * np.log(1 - p))),
        "brier": float(np.mean((xp - completed) ** 2)),
        "slope": float(model.coef_[0][0]),
        "slope_intercept": float(model.intercept_[0]),
    }


def run_variant(cfg: GenConfig, seeds: list[int], scorer: XPassScorer) -> dict[str, Any]:
    matches = [simulate(s, cfg, scorer) for s in seeds]
    cols = {k: np.concatenate([m.cols[k] for m in matches]) for k in matches[0].cols}
    completed = np.concatenate([m.completed for m in matches])
    xp = np.concatenate([m.xp for m in matches])
    terms = {t: np.concatenate([m.terms[t] for m in matches]) for t in TERMS}
    seed_gaps = np.array([m.completed.mean() - m.xp.mean() for m in matches])
    labels = assign_buckets(cols)
    return {
        "passes": int(len(completed)),
        "passes_per_match": float(len(completed) / len(seeds)),
        "completion": float(completed.mean()),
        "xp": float(xp.mean()),
        "gap": float(completed.mean() - xp.mean()),
        "gap_seed_std": float(seed_gaps.std(ddof=1)) if len(seeds) > 1 else 0.0,
        "synthetic_buckets": bucket_rates(labels, completed),
        "xp_buckets": bucket_rates(labels, xp),
        "mean_length": float(cols["length"].mean()),
        "share_high": float(cols["is_high"].mean()),
        "share_low": float(cols["is_low"].mean()),
        "share_under_pressure": float(cols["under_pressure"].mean()),
        "share_final_third": float((cols["start_x"] >= FINAL_THIRD_X).mean()),
        "calibration": calibration(xp, completed),
        # raw arrays for the reliability table and the diagnosis
        "_cols": cols, "_completed": completed, "_xp": xp, "_terms": terms,
    }


MIN_SYNTH_N = 200


def bucket_mad(real: dict, synthetic: dict) -> dict[str, float | int]:
    """|synthetic - real| over every bucket both sides have passes in.

    mad: plain mean; weighted: weighted by the bucket's real pass count (each dimension
    covers every real pass once, so each dimension carries the same total weight);
    mad_min_n: plain mean over buckets with at least MIN_SYNTH_N synthetic passes.
    """
    cells = [
        (abs(synthetic[dim][b]["rate"] - real[dim][b]["rate"]), real[dim][b]["n"], synthetic[dim][b]["n"])
        for dim, names in BUCKETS.items() for b in names
        if synthetic[dim][b]["rate"] is not None and real[dim][b]["rate"] is not None
    ]
    diffs = np.array([c[0] for c in cells])
    real_n = np.array([c[1] for c in cells], dtype=float)
    big = np.array([c[2] >= MIN_SYNTH_N for c in cells])
    return {
        "mad": float(diffs.mean()),
        "weighted": float(np.sum(diffs * real_n) / real_n.sum()),
        "mad_min_n": float(diffs[big].mean()) if big.any() else float("nan"),
        "cells": len(cells),
        "cells_min_n": int(big.sum()),
    }


def reliability_rows(xp: np.ndarray, completed: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """Per equal-width xP bin: passes, mean xP, actual completion and actual - xP."""
    idx = np.minimum((xp * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = idx == i
        n = int(mask.sum())
        rows.append({
            "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
            "passes": n,
            "mean_xp": float(xp[mask].mean()) if n else None,
            "actual": float(completed[mask].mean()) if n else None,
            "diff": float(completed[mask].mean() - xp[mask].mean()) if n else None,
        })
    rows.append({"bin": "all", "passes": int(len(xp)), "mean_xp": float(xp.mean()),
                 "actual": float(completed.mean()), "diff": float(completed.mean() - xp.mean())})
    return rows


def print_reliability(rows: list[dict[str, Any]]) -> None:
    print(f"{'xP bin':<10}{'passes':>8}{'mean xP %':>11}{'actual %':>10}{'diff pp':>9}")
    for r in rows:
        diff = f"{100 * r['diff']:>9.1f}" if r["diff"] is not None else f"{'-':>9}"
        print(f"{r['bin']:<10}{r['passes']:>8,}{_pct(r['mean_xp']):>11}{_pct(r['actual']):>10}{diff}")


def _pct(v: float | None) -> str:
    return f"{100 * v:6.1f}" if v is not None else f"{'-':>6}"


def print_buckets(name: str, res: dict[str, Any], real: dict[str, Any], summary: dict[str, Any]) -> None:
    print(f"\n=== {name}: {res['passes']:,} synthetic passes ===")
    print(f"{'bucket':<24}{'real %':>8}{'synth %':>9}{'xP %':>8}{'synth n':>9}")
    for dim, names in BUCKETS.items():
        for b in names:
            s, x = res["synthetic_buckets"][dim][b], res["xp_buckets"][dim][b]
            print(f"{dim + ' ' + b:<24}{_pct(real[dim][b]['rate']):>8}{_pct(s['rate']):>9}{_pct(x['rate']):>8}{s['n']:>9}")
    print(f"\n{'pass mix':<28}{'real':>8}{'synth':>9}")
    print(f"{'mean pass length':<28}{summary['mean_length']:>8.1f}{res['mean_length']:>9.1f}")
    for key, label in (("share_high", "share high %"), ("share_low", "share low %"),
                       ("share_under_pressure", "share under pressure %"),
                       ("share_final_third", f"share start_x >= {FINAL_THIRD_X:.0f} %")):
        print(f"{label:<28}{_pct(summary.get(key)):>8}{_pct(res[key]):>9}")
    real_ppm = summary.get("passes_per_match")
    print(f"{'passes per match':<28}{real_ppm if real_ppm is None else round(real_ppm, 1)!s:>8}"
          f"{res['passes_per_match']:>9.1f}")


def print_calibration(name: str, res: dict[str, Any]) -> None:
    c = res["calibration"]
    print(f"\n=== calibration of xP on {name} ({res['passes']:,} passes) ===")
    print(f"actual {_pct(res['completion']).strip()}%  mean xP {_pct(res['xp']).strip()}%  "
          f"gap {100 * res['gap']:.2f} pp")
    print_reliability(reliability_rows(res["_xp"], res["_completed"]))
    print(f"log-loss {c['log_loss']:.4f}  Brier {c['brier']:.4f}  calibration slope {c['slope']:.3f} "
          f"(intercept {c['slope_intercept']:.3f})")


def diagnose(res: dict[str, Any]) -> None:
    xp, completed, cols, terms = res["_xp"], res["_completed"], res["_cols"], res["_terms"]
    groups = {"xP < 0.4": xp < 0.4, "xP > 0.8": xp > 0.8}
    logit = sum(terms[t] for t in TERMS)
    true_p = 1 / (1 + np.exp(-logit))

    print(f"\n{'':<26}" + "".join(f"{g:>12}" for g in groups))
    rows: list[tuple[str, Any]] = [
        ("passes", lambda m: f"{int(m.sum()):>12,}"),
        ("mean xP %", lambda m: f"{_pct(xp[m].mean()):>12}"),
        ("actual completion %", lambda m: f"{_pct(completed[m].mean()):>12}"),
        ("generator p (mean) %", lambda m: f"{_pct(true_p[m].mean()):>12}"),
        ("share high %", lambda m: f"{_pct(cols['is_high'][m].mean()):>12}"),
        ("share low %", lambda m: f"{_pct(cols['is_low'][m].mean()):>12}"),
        ("share under pressure %", lambda m: f"{_pct(cols['under_pressure'][m].mean()):>12}"),
        ("share headers %", lambda m: f"{_pct(cols['is_head'][m].mean()):>12}"),
        ("share corner %", lambda m: f"{_pct(cols['corner'][m].mean()):>12}"),
        ("share goal kick %", lambda m: f"{_pct(cols['goal_kick'][m].mean()):>12}"),
        ("share throw-in %", lambda m: f"{_pct(cols['throw_in'][m].mean()):>12}"),
        ("mean length", lambda m: f"{cols['length'][m].mean():>12.1f}"),
        ("mean forward_gain", lambda m: f"{cols['forward_gain'][m].mean():>12.1f}"),
        ("share start_x >= 90 %", lambda m: f"{_pct((cols['start_x'][m] >= 90).mean()):>12}"),
    ]
    for label, fmt in rows:
        print(f"{label:<26}" + "".join(fmt(m) for m in groups.values()))

    print(f"\nmean contribution to the generator logit")
    print(f"{'term':<26}" + "".join(f"{g:>12}" for g in groups) + f"{'difference':>12}")
    for t in (*TERMS, "total"):
        vals = [float((logit if t == "total" else terms[t])[m].mean()) for m in groups.values()]
        print(f"{t:<26}" + "".join(f"{v:>12.3f}" for v in vals) + f"{vals[0] - vals[1]:>12.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare fitted completion weights inside the generator.")
    parser.add_argument("--fit", type=Path, default=ROOT / "reports" / "outcome_fit.json")
    parser.add_argument("--seeds", type=int, default=20, help="generate seeds 1..N per variant")
    parser.add_argument("--reliability", action="store_true",
                        help="print a reliability table for the current GenConfig defaults and exit")
    parser.add_argument("--diagnose", action="store_true",
                        help="pass mix and logit terms of hard vs easy passes for the current defaults")
    parser.add_argument("--calibrate", nargs="*", default=["C", "D"],
                        help="variants to print calibration of xP for (default: C D)")
    parser.add_argument("--check-seeds", nargs=2, type=int, default=[21, 40], metavar=("FIRST", "LAST"),
                        help="second seed range for an out-of-sample summary of the --calibrate variants")
    parser.add_argument("--buckets", nargs="*", default=["D"],
                        help="variants to print bucket tables and pass mix for (default: D)")
    args = parser.parse_args(argv)
    seeds = list(range(1, args.seeds + 1))

    if args.reliability or args.diagnose:
        scorer = XPassScorer(*find_default_model())
        res = run_variant(GenConfig(), seeds, scorer)
        print(f"GenConfig defaults, seeds 1..{args.seeds}, model {scorer.model_version}")
        if args.reliability:
            print_reliability(reliability_rows(res["_xp"], res["_completed"]))
        if args.diagnose:
            diagnose(res)
        return 0

    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    real, summary = fit["real_buckets"], fit["summary"]
    configs = {"H": variant_config(fit["hand_set"]),
               **{v: variant_config(fit[k]["coefficients"]) for v, k in VARIANT_KEYS.items() if k in fit}}
    scorer = XPassScorer(*find_default_model())

    results = {name: run_variant(cfg, seeds, scorer) for name, cfg in configs.items()}

    print(f"seeds 1..{args.seeds}, model {scorer.model_version}, real completion {_pct(summary['completion_rate']).strip()}%")
    print(f"\n{'variant':<9}{'passes':>8}{'actual %':>10}{'xP %':>8}{'gap pp':>8}{'seed std pp':>13}"
          f"{'log-loss':>10}{'Brier':>8}{'slope':>7}")
    for name, r in results.items():
        c = r["calibration"]
        print(f"{name:<9}{r['passes']:>8,}{_pct(r['completion']):>10}{_pct(r['xp']):>8}"
              f"{100 * r['gap']:>8.2f}{100 * r['gap_seed_std']:>13.2f}"
              f"{c['log_loss']:>10.4f}{c['brier']:>8.4f}{c['slope']:>7.3f}")
    print("real hold-out xP reference: log-loss 0.4007, Brier 0.1266")

    print(f"\nMAD synthetic vs real bucket rates (pp)")
    print(f"{'variant':<9}{'plain':>8}{'weighted':>10}{f'n>={MIN_SYNTH_N}':>9}{'cells':>7}{f'cells n>={MIN_SYNTH_N}':>15}")
    for name, r in results.items():
        m = bucket_mad(real, r["synthetic_buckets"])
        print(f"{name:<9}{100 * m['mad']:>8.2f}{100 * m['weighted']:>10.2f}{100 * m['mad_min_n']:>9.2f}"
              f"{m['cells']:>7}{m['cells_min_n']:>15}")

    for name in args.calibrate:
        if name in results:
            print_calibration(name, results[name])

    first, last = args.check_seeds
    check_seeds = list(range(first, last + 1))
    print(f"\n=== out-of-sample generator seeds {first}..{last} ===")
    print(f"{'variant':<9}{'seeds':>8}{'passes':>8}{'actual %':>10}{'xP %':>8}{'gap pp':>8}"
          f"{'slope':>7}{'log-loss':>10}{'Brier':>8}")
    for name in args.calibrate:
        if name not in configs:
            continue
        for label, r in ((f"{1}..{args.seeds}", results[name]),
                         (f"{first}..{last}", run_variant(configs[name], check_seeds, scorer))):
            c = r["calibration"]
            print(f"{name:<9}{label:>8}{r['passes']:>8,}{_pct(r['completion']):>10}{_pct(r['xp']):>8}"
                  f"{100 * r['gap']:>8.2f}{c['slope']:>7.3f}{c['log_loss']:>10.4f}{c['brier']:>8.4f}")
    for name in args.buckets:
        if name in results:
            print_buckets(name, results[name], real, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
