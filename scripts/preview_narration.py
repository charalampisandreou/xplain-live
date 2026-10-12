"""
Preview the Narrator on real moments: seed 42, the first N commentated moments (qualifies_live),
narrated by the Foundry model and checked by the Verifier. Needs `az login` (AzureCliCredential).

--spread picks 8 moments for coverage instead: per half, the top commentated moment of each kind
(select_half_highlights) and the half's last commentated moment; the first moment in added time;
and one moment with a default-condition label (prompts.DEFAULT_LABELS) as driver 1 or 2, preferring
one with fewer than 3 drivers. Commentated moments are preferred; otherwise any moment is used.

Prints, per moment: the card fields the UI shows from code (minute_label, xP percent), the
difficulty_word and the facts, the three lines, drivers_used, attempts, the Verifier reasons and notes,
and the latency; then how many were accepted first time, after the retry, or rejected, and how many
accepted lines had no difficulty word.

Usage:
    python scripts/preview_narration.py [--file data/match_seed42.jsonl] [--n 8] [--spread]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from azure.identity.aio import AzureCliCredential  # noqa: E402

from xplain.analyst import OUTLIER_KINDS, select_half_highlights  # noqa: E402
from xplain.analytics import percent  # noqa: E402
from xplain.contracts import MomentBundle  # noqa: E402
from xplain.narrator import Narrator, build_agent, build_facts  # noqa: E402
from xplain.pipeline import ensure_match_file, load_events, process_match  # noqa: E402
from xplain.prompts import DEFAULT_LABELS  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "match_seed42.jsonl"
HALF_NAMES = {1: "first half", 2: "second half"}


def _first(candidates: list[MomentBundle]) -> MomentBundle | None:
    """The first candidate in match order, commentated moments before the others."""
    ordered = sorted(candidates, key=lambda b: not b.qualifies_live)
    return ordered[0] if ordered else None


def _pool(b: MomentBundle) -> str:
    return "commentated" if b.qualifies_live else "not commentated (no commentated moment qualifies)"


def spread(bundles: list[MomentBundle]) -> list[tuple[str, MomentBundle]]:
    """Up to 8 (why, bundle) pairs chosen for coverage, in match order, no moment twice."""
    live = [b for b in bundles if b.qualifies_live]
    by_id = {b.moment.moment_id: b for b in bundles}
    picks: dict[str, tuple[str, MomentBundle]] = {}

    def add(why: str, b: MomentBundle | None) -> None:
        if b is not None and b.moment.moment_id not in picks:
            picks[b.moment.moment_id] = (why, b)

    top = select_half_highlights([b.moment for b in live], per_kind=1)
    for period, half in HALF_NAMES.items():
        for kind in OUTLIER_KINDS:
            for m in top[period][kind]:
                add(f"{half}: top commentated {kind}", by_id[m.moment_id])
        in_half = [b for b in live if b.moment.period == period and b.moment.moment_id not in picks]
        add(f"{half}: last commentated moment of the half", in_half[-1] if in_half else None)

    added = _first([b for b in bundles if b.moment.minute_added > 0])
    if added is not None:
        add(f"added time ({added.moment.minute_label}), {_pool(added)}", added)

    with_default = [b for b in bundles
                    if any(d.label in DEFAULT_LABELS for d in b.explanation.drivers[:2])]
    fewer = [b for b in with_default if len(b.explanation.drivers) < 3]
    chosen = _first(fewer) or _first(with_default)
    if chosen is not None:
        labels = [d.label for d in chosen.explanation.drivers[:2] if d.label in DEFAULT_LABELS]
        size = f"{len(chosen.explanation.drivers)} drivers" + ("" if fewer else "; no moment has fewer than 3")
        add(f"default label {labels} as driver 1 or 2 ({size}), {_pool(chosen)}", chosen)

    return sorted(picks.values(), key=lambda p: p[1].moment.seq)


async def main(path: Path, n: int, use_spread: bool) -> None:
    ensure_match_file(path, 42)
    bundles = process_match(load_events(path), XPassScorer(*find_default_model()))
    if use_spread:
        selected = spread(bundles)
    else:
        selected = [("commentated, in match order", b) for b in bundles if b.qualifies_live][:n]
    results = []
    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential))
        for i, (why, bundle) in enumerate(selected, start=1):
            result = await narrator.narrate(bundle)
            results.append(result)
            print(f"=== {i}. {bundle.moment.moment_id} ===")
            facts = build_facts(bundle)
            print("selected:", why)
            print(f"card: minute {bundle.moment.minute_label} | xP {percent(bundle.moment.xp)}% (from code)")
            print("difficulty_word:", json.dumps(facts["difficulty_word"], ensure_ascii=False))
            print("facts:", json.dumps(facts, ensure_ascii=False))
            if result.line is not None:
                print("EN:", result.line.line_en)
                print("ES:", result.line.line_es)
                print("EL:", result.line.line_el)
                print("drivers_used:", result.line.drivers_used)
            else:
                print("no line")
            print(f"attempts: {result.attempts} | latency: {result.latency_s:.2f} s")
            print("reasons:", result.reasons or "none")
            print("notes:", result.notes or "none")
            print()

    first = sum(r.line is not None and r.attempts == 1 for r in results)
    retry = sum(r.line is not None and r.attempts > 1 for r in results)
    rejected = sum(r.line is None for r in results)
    mean = sum(r.latency_s for r in results) / len(results) if results else 0.0
    no_word = sum(len(r.notes) for r in results if r.line is not None)
    lines = 3 * sum(r.line is not None for r in results)
    print(f"accepted first try {first}, after retry {retry}, rejected {rejected}; "
          f"mean latency {mean:.2f} s per moment")
    print(f"accepted lines with no difficulty word: {no_word} of {lines}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Narrate the first commentated moments of a match.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--n", type=int, default=8, help="number of moments without --spread")
    parser.add_argument("--spread", action="store_true", help="pick 8 moments for coverage instead of the first N")
    args = parser.parse_args()
    asyncio.run(main(args.file, args.n, args.spread))
