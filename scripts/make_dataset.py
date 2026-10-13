"""
Build the xPass training and hold-out tables from generated matches.

Train: the first --train-matches seeds from 1, skipping 42 (default 250: seeds 1..251 without 42).
Hold-out: --holdout-matches seeds from 2001 (default 100: seeds 2001..2100). Seed 42 is the demo match and is
in neither set. Each row is one pass: the 17 FEATURE_ORDER columns (built by
xplain.features from PassInputs only, so no end-derived field), completed (0/1) and match_id.

Usage:
    python scripts/make_dataset.py [--out data] [--train-matches 250] [--holdout-matches 100]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xplain.generator import generate_match  # noqa: E402
from xplain.train import pass_table  # noqa: E402

DEMO_SEED = 42
HOLDOUT_START = 2001


def train_seeds(n: int) -> tuple[int, ...]:
    """The first n seeds from 1, skipping the demo seed."""
    return tuple(s for s in range(1, n + 2) if s != DEMO_SEED)[:n]


def holdout_seeds(n: int) -> tuple[int, ...]:
    return tuple(range(HOLDOUT_START, HOLDOUT_START + n))


def build(seeds: tuple[int, ...]) -> pd.DataFrame:
    return pd.concat([pass_table(generate_match(seed)[0]) for seed in seeds], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matches and write the xPass train / hold-out tables.")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--train-matches", type=int, default=250)
    parser.add_argument("--holdout-matches", type=int, default=100)
    args = parser.parse_args()
    train, holdout = train_seeds(args.train_matches), holdout_seeds(args.holdout_matches)
    assert DEMO_SEED not in train and DEMO_SEED not in holdout
    assert not set(train) & set(holdout), "train seeds reach the hold-out range"
    assert len(train) == args.train_matches and len(holdout) == args.holdout_matches

    args.out.mkdir(parents=True, exist_ok=True)
    for name, seeds in (("train", train), ("holdout", holdout)):
        start = time.perf_counter()
        table = build(seeds)
        path = args.out / f"xpass_{name}.csv"
        table.to_csv(path, index=False)
        print(f"{name}: seeds {seeds[0]}-{seeds[-1]}, {len(seeds)} matches, {len(table)} rows, "
              f"completion {table['completed'].mean():.4f}, {time.perf_counter() - start:.1f} s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
