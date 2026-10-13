"""
Base-rate check for the outlier thresholds. Offline: generated matches, the default xPass model.

For passes at or above the failure threshold (xP >= surprise_failure_min_xp) it prints the expected
number of failures, sum(1 - xP), next to the observed failures; for passes at or below the success
threshold (xP <= surprise_success_max_xp), the expected successes, sum(xP), next to the observed ones.
The thresholds are read from analytics.DEFAULT; this script never changes them.

Usage:
    python scripts/base_rate_check.py [--seeds 1001-1010]
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xplain import analytics  # noqa: E402
from xplain.config import AnalyticsConfig  # noqa: E402
from xplain.generator import generate_match  # noqa: E402
from xplain.schema import PassEvent  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402


def base_rates(xps: Sequence[float], completed: Sequence[bool], cfg: AnalyticsConfig = analytics.DEFAULT) -> dict:
    """Expected and observed counts in the failure zone (high xP) and the success zone (low xP)."""
    high = [(x, c) for x, c in zip(xps, completed, strict=True) if x >= cfg.surprise_failure_min_xp]
    low = [(x, c) for x, c in zip(xps, completed, strict=True) if x <= cfg.surprise_success_max_xp]
    return {
        "passes": len(xps),
        "failure_zone": {"threshold": cfg.surprise_failure_min_xp, "passes": len(high),
                         "expected_failures": sum(1 - x for x, _ in high),
                         "observed_failures": sum(not c for _, c in high)},
        "success_zone": {"threshold": cfg.surprise_success_max_xp, "passes": len(low),
                         "expected_successes": sum(x for x, _ in low),
                         "observed_successes": sum(bool(c) for _, c in low)},
    }


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expected vs observed outcomes beyond the outlier thresholds.")
    parser.add_argument("--seeds", default="1001-1010", help="seed range, e.g. 1001-1010")
    args = parser.parse_args(argv)
    scorer = XPassScorer(*find_default_model())
    xps: list[float] = []
    completed: list[bool] = []
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        passes = [e for e in generate_match(seed)[0] if isinstance(e, PassEvent)]
        xps += [r.expected_probability for r in scorer.score([p.inputs for p in passes])]
        completed += [p.details.completed for p in passes]
    r = base_rates(xps, completed)
    f, s = r["failure_zone"], r["success_zone"]
    print(f"model {scorer.model_version}, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} matches), {r['passes']} passes")
    print(f"xP >= {f['threshold']}: {f['passes']} passes, expected failures sum(1-xP) = {f['expected_failures']:.1f}, "
          f"observed failures = {f['observed_failures']}")
    print(f"xP <= {s['threshold']}: {s['passes']} passes, expected successes sum(xP) = {s['expected_successes']:.1f}, "
          f"observed successes = {s['observed_successes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
