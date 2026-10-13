"""
Which narrated driver labels occur together in commentated moments. Offline: generated matches, the
default xPass model; no LLM.

For each seed, every commentated moment (qualifies_live) is taken, and its mentionable drivers kept:
the label is not in phrases.NOT_NARRATED, and the direction matches the moment's difficulty tier
("raises" for easy, "lowers" for hard), as the Narrator may only mention those. INCOMPATIBLE_PAIRS is
NOT applied, so the table shows what the filter would act on; pairs already in it are marked "[listed]".

Prints the co-occurring label pairs with counts, most frequent first, then the pairs that never occur
among the labels that do occur (two labels of the same input, e.g. two heights, cannot occur together
and are not listed).

Usage:
    python scripts/driver_pairs.py [--seeds 1-50]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xplain.contracts import Driver, MomentBundle  # noqa: E402
from xplain.generator import generate_match  # noqa: E402
from xplain.narration import difficulty_tier  # noqa: E402
from xplain.phrases import INCOMPATIBLE_PAIRS, is_narrated  # noqa: E402
from xplain.pipeline import process_match  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def mentionable(bundle: MomentBundle) -> list[Driver]:
    """Narrated drivers whose direction matches the moment's tier, in rank order."""
    wanted = "raises" if difficulty_tier(bundle.moment).family == "easy" else "lowers"
    return [d for d in bundle.explanation.drivers if is_narrated(d) and d.direction == wanted]


def pair_counts(bundles: list[MomentBundle]) -> tuple[Counter, dict[str, str]]:
    """(Counter of frozenset label pairs, {label: feature}) over the commentated moments."""
    pairs: Counter = Counter()
    feature: dict[str, str] = {}
    for b in bundles:
        if not b.qualifies_live:
            continue
        drivers = mentionable(b)
        feature.update({d.label: d.feature for d in drivers})
        pairs.update(frozenset({a.label, c.label}) for a, c in itertools.combinations(drivers, 2))
    return pairs, feature


def _pair(p: frozenset) -> str:
    a, b = sorted(p)
    return f"{a} + {b}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Co-occurring narrated driver labels in commentated moments.")
    parser.add_argument("--seeds", default="1-50", help="seed range, e.g. 1-50")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    scorer = XPassScorer(*find_default_model())
    bundles: list[MomentBundle] = []
    for seed in seeds:
        bundles += process_match(generate_match(seed)[0], scorer)
    live = [b for b in bundles if b.qualifies_live]
    pairs, feature = pair_counts(bundles)
    sizes = Counter(len(mentionable(b)) for b in live)

    print(f"seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} matches), {len(live)} commentated moments")
    print("mentionable drivers per moment: " + ", ".join(f"{k}: {sizes[k]}" for k in sorted(sizes)))
    print()
    print("co-occurring pairs (count, pair), most frequent first:")
    for p, n in sorted(pairs.items(), key=lambda kv: (-kv[1], _pair(kv[0]))):
        print(f"{n:5}  {_pair(p)}{'  [listed]' if p in INCOMPATIBLE_PAIRS else ''}")
    print()
    labels = sorted(feature)
    never = [frozenset({a, b}) for a, b in itertools.combinations(labels, 2)
             if feature[a] != feature[b] and frozenset({a, b}) not in pairs]
    print(f"pairs that never occur ({len(never)}), among the {len(labels)} labels that do occur:")
    for p in sorted(never, key=_pair):
        print(f"       {_pair(p)}{'  [listed]' if p in INCOMPATIBLE_PAIRS else ''}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
