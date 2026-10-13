"""
Preview the Narrator on real moments: seed 42, the first N commentated moments (qualifies_live),
narrated by the Foundry model and checked by the Verifier. Needs `az login` (AzureCliCredential).

--spread picks up to 9 moments for coverage instead. All but one have at least one narrated driver
(after the phrases.NOT_NARRATED filter): per half, the top commentated moment of each kind
(select_half_highlights) and the half's last commentated moment; the first moment in added time; and
one moment with a default-condition label (prompts.DEFAULT_LABELS) as driver 1 or 2, preferring one
with fewer than 3 drivers. The last pick is one moment with zero narrated drivers. Commentated moments
are preferred; otherwise any moment is used.

Prints the phrase mode first (config.phrase_mode, set with XPLAIN_PHRASE_MODE), then per moment: the
card fields the UI shows from code (minute_label, xP percent), the difficulty_word and the facts, the
narrated drivers (rank, direction, label or phrase), the three lines as plain text, one language per
line, drivers_used, attempts, the Verifier reasons with each rejected candidate's three lines, the notes
and the latency; then how many were accepted first time, after the retry, or rejected, and how
many accepted lines had no difficulty word, after the template variety: each player's name masked as X,
"distinct EN templates: X of N" with the three most repeated templates and their counts, then the same
for ES.

--all narrates every commentated moment of the match instead. --seed N uses data/match_seed<N>.jsonl
(generated if missing). --mode strict|free|hints overrides XPLAIN_PHRASE_MODE, so the same moments can
be run in two modes back to back.

Usage:
    python scripts/preview_narration.py [--file PATH | --seed N] [--n 8 | --spread | --all]
                                        [--mode strict|free|hints]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from azure.identity.aio import AzureCliCredential  # noqa: E402

from xplain.analyst import OUTLIER_KINDS, select_half_highlights  # noqa: E402
from xplain.analytics import percent  # noqa: E402
from xplain.config import DEFAULT_PHRASE_MODE, PHRASE_MODES, phrase_mode  # noqa: E402
from xplain.contracts import MomentBundle  # noqa: E402
from xplain.narrator import Narrator, build_agent, build_facts  # noqa: E402
from xplain.phrases import driver_phrases  # noqa: E402
from xplain.pipeline import ensure_match_file, load_events, process_match  # noqa: E402
from xplain.prompts import DEFAULT_LABELS  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402
from xplain.verifier import narrated_drivers  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "match_seed42.jsonl"
HALF_NAMES = {1: "first half", 2: "second half"}


def _first(candidates: list[MomentBundle]) -> MomentBundle | None:
    """The first candidate in match order, commentated moments before the others."""
    ordered = sorted(candidates, key=lambda b: not b.qualifies_live)
    return ordered[0] if ordered else None


def _pool(b: MomentBundle) -> str:
    return "commentated" if b.qualifies_live else "not commentated (no commentated moment qualifies)"


def _driver_notes(b: MomentBundle, lang: str, mode: str) -> str:
    """The narrated drivers of a moment: rank, direction and the English label (free and hints modes), or
    the approved phrase in that language (strict mode)."""
    drivers = narrated_drivers(b)
    if not drivers:
        return "[no narrated drivers]"
    text = (lambda d: d.label) if mode != "strict" else (lambda d: driver_phrases(d)["phrase"][lang])
    return "[" + "; ".join(f"{rank} {d.direction} {text(d)!r}" for rank, d in drivers.items()) + "]"


def spread(all_bundles: list[MomentBundle]) -> list[tuple[str, MomentBundle]]:
    """Up to 9 (why, bundle) pairs chosen for coverage, in match order, no moment twice: up to 8 with a
    narrated driver, and one with none."""
    bundles = [b for b in all_bundles if narrated_drivers(b)]
    by_id = {b.moment.moment_id: b for b in bundles}
    live = [b for b in bundles if b.qualifies_live]
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

    bare = _first([b for b in all_bundles if not narrated_drivers(b)])
    if bare is not None:
        add(f"zero narrated drivers ({len(bare.explanation.drivers)} drivers, all dropped), {_pool(bare)}", bare)
    else:
        print("note: no moment in this match has zero narrated drivers")

    return sorted(picks.values(), key=lambda p: p[1].moment.seq)


def template(line: str, player: str | None) -> str:
    """The line with the player's name replaced by X, so lines that differ only in the name compare equal."""
    return line.replace(player, "X") if player else line


def template_variety(lines: list[tuple[str, str | None]], lang: str) -> None:
    """Print how many distinct templates the accepted lines of one language have, and the most repeated."""
    counts = Counter(template(line, player) for line, player in lines)
    print(f"distinct {lang} templates: {len(counts)} of {len(lines)}")
    for text, n in counts.most_common(3):
        print(f"  {n}x  {text}")


async def main(path: Path, seed: int, n: int, how: str, mode: str) -> None:
    ensure_match_file(path, seed)
    bundles = process_match(load_events(path), XPassScorer(*find_default_model()))
    commentated = [("commentated, in match order", b) for b in bundles if b.qualifies_live]
    selected = {"spread": spread(bundles), "all": commentated, "first": commentated[:n]}[how]
    print(f"PHRASE_MODE: {mode} (choose with --mode or XPLAIN_PHRASE_MODE)")
    print(f"match: {path} | moments: {how} ({len(selected)})")
    print()
    results = []
    players: list[str | None] = []
    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential, mode=mode), mode=mode)
        for i, (why, bundle) in enumerate(selected, start=1):
            result = await narrator.narrate(bundle)
            results.append(result)
            players.append(bundle.moment.player_name)
            print(f"=== {i}. {bundle.moment.moment_id} ===")
            facts = build_facts(bundle, mode)
            print("selected:", why)
            print(f"card: minute {bundle.moment.minute_label} | xP {percent(bundle.moment.xp)}% (from code)")
            print("difficulty_word:", json.dumps(facts["difficulty_word"], ensure_ascii=False))
            print("facts:", json.dumps(facts, ensure_ascii=False))
            print("narrated drivers:", _driver_notes(bundle, "en", mode))
            if result.line is not None:
                for lang in ("en", "es", "el"):   # plain text, one language per line, easy to copy
                    print(f"{lang.upper()}: {getattr(result.line, f'line_{lang}')}")
                print("drivers_used:", result.line.drivers_used)
            else:
                print("no line")
            print(f"attempts: {result.attempts} | latency: {result.latency_s:.2f} s")
            print("reasons:", result.reasons or "none")
            for k, cand in enumerate(result.rejected, start=1):   # diagnostics only, never shown in the UI
                print(f"rejected candidate {k} (drivers_used {cand.drivers_used}):")
                for lang in ("en", "es", "el"):
                    print(f"{lang.upper()}: {getattr(cand, f'line_{lang}')}")
            print("notes:", result.notes or "none")
            print()

    accepted = [r.line for r in results if r.line is not None]
    first = sum(r.line is not None and r.attempts == 1 for r in results)
    retry = sum(r.line is not None and r.attempts > 1 for r in results)
    rejected = sum(r.line is None for r in results)
    mean = sum(r.latency_s for r in results) / len(results) if results else 0.0
    no_word = sum(len(r.notes) for r in results if r.line is not None)
    named = [(r.line, player) for r, player in zip(results, players, strict=True) if r.line is not None]
    template_variety([(line.line_en, player) for line, player in named], "EN")
    template_variety([(line.line_es, player) for line, player in named], "ES")
    print(f"accepted first try {first}, after retry {retry}, rejected {rejected}; "
          f"mean latency {mean:.2f} s per moment")
    print(f"accepted lines with no difficulty word: {no_word} of {3 * len(accepted)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Narrate commentated moments of a match.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--file", type=Path, help="match file (generated from seed 42 if missing)")
    where.add_argument("--seed", type=int, help="use data/match_seed<N>.jsonl, generating it if missing")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--spread", action="store_true", help="pick up to 9 moments for coverage")
    which.add_argument("--all", action="store_true", help="every commentated moment of the match")
    parser.add_argument("--n", type=int, default=8, help="number of moments without --spread or --all")
    parser.add_argument("--mode", choices=PHRASE_MODES, help="phrase mode (default: XPLAIN_PHRASE_MODE or "
                                                             f"{DEFAULT_PHRASE_MODE})")
    args = parser.parse_args()
    seed = 42 if args.seed is None else args.seed
    path = args.file or (DEFAULT_FILE if args.seed is None else ROOT / "data" / f"match_seed{seed}.jsonl")
    how = "spread" if args.spread else "all" if args.all else "first"
    asyncio.run(main(path, seed, args.n, how, args.mode or phrase_mode()))
