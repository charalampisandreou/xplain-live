"""
Build the xPass training and hold-out tables from generated matches.

Train: seeds 1-201 without 42 (200 matches). Hold-out: seeds 1001-1050 (50 matches). Seed 42 is the
demo match and is in neither set. Each row is one pass: the 17 FEATURE_ORDER columns (built by
xplain.features from PassInputs only, so no end-derived field), completed (0/1) and match_id.

Usage:
    python scripts/make_dataset.py [--out data]
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
TRAIN_SEEDS = tuple(s for s in range(1, 202) if s != DEMO_SEED)
HOLDOUT_SEEDS = tuple(range(1001, 1051))


def build(seeds: tuple[int, ...]) -> pd.DataFrame:
    return pd.concat([pass_table(generate_match(seed)[0]) for seed in seeds], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matches and write the xPass train / hold-out tables.")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    assert DEMO_SEED not in TRAIN_SEEDS and DEMO_SEED not in HOLDOUT_SEEDS
    assert not set(TRAIN_SEEDS) & set(HOLDOUT_SEEDS) and len(TRAIN_SEEDS) == 200

    args.out.mkdir(parents=True, exist_ok=True)
    for name, seeds in (("train", TRAIN_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        start = time.perf_counter()
        table = build(seeds)
        path = args.out / f"xpass_{name}.csv"
        table.to_csv(path, index=False)
        print(f"{name}: {len(seeds)} matches, {len(table)} passes, completion {table['completed'].mean():.4f}, "
              f"{time.perf_counter() - start:.1f} s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
