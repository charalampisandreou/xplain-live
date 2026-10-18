"""
Preview one recap with the real RecapEditor model (needs az login). Prints the card, the EN/ES/EL text and the trace;
writes nothing (no bundle).

    python -u scripts/preview_recap.py [--file data/match_seed3001.jsonl] [--scope full_time|half_time]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from xplain.recap import RECAP_SCOPES, RecapCard
from xplain.recap_editor import RecapResult


def card_lines(card: RecapCard) -> list[str]:
    """The card as text: score, goals with minute_label, each team's passing (every number from code)."""
    rows = [f"{card.scope}: {card.home.name} {card.score_home}-{card.score_away} {card.away.name} ({card.result})"]
    rows += [f"  goal {g.minute_label}' {g.team_name}: {g.player_name or '-'}{' (late)' if g.late else ''}"
             for g in card.goals]
    for t in (card.home, card.away):
        rows.append(f"  {t.name}: {t.completed}/{t.passes} completed, expected {t.expected:.1f}, {t.label} "
                    f"({t.band})")
    return rows


def result_lines(r: RecapResult) -> list[str]:
    rows = [f"status {r.status}, {r.attempts} attempt(s), {r.latency_s:.1f} s"]
    if r.text is not None:
        rows += [f"  EN: {r.text.text_en}", f"  ES: {r.text.text_es}", f"  EL: {r.text.text_el}",
                 f"  claims_used: {r.text.claims_used}"]
    rows += [f"  reason: {x}" for x in r.reasons] + [f"  note: {x}" for x in r.notes]
    for i, c in enumerate(r.rejected, start=1):
        rows += [f"  rejected candidate {i}:", f"    EN: {c.text_en}", f"    ES: {c.text_es}", f"    EL: {c.text_el}",
                 f"    claims_used: {c.claims_used}"]
    rows += ["trace:"] + [f"  {s.agent} ({s.kind}) attempt {s.attempt}: {'pass' if s.ok else 'fail'}"
                          + (f", {s.model}, {s.latency_s:.2f} s" if s.kind == "llm" else "")
                          + (f" | {'; '.join(s.reasons)}" if s.reasons else "") for s in r.trace]
    return rows


async def _run(path: Path, scope: str) -> None:
    from azure.identity.aio import AzureCliCredential

    from xplain.bundle import _load_scorer
    from xplain.pipeline import load_events, pass_marks
    from xplain.recap import recap_card, roster
    from xplain.recap_editor import RecapEditor, build_recap_agent

    events = load_events(path)
    card = recap_card(events, pass_marks(events, _load_scorer()), scope)
    print("\n".join(card_lines(card)), flush=True)
    async with AzureCliCredential() as credential:
        editor = RecapEditor(build_recap_agent(credential))
        print(f"model: {editor.model}", flush=True)
        result = await editor.write(card, roster(events))
    print("\n".join(result_lines(result)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one recap with the real model and print it (no bundle).")
    parser.add_argument("--file", type=Path, default=Path("data/match_seed3001.jsonl"))
    parser.add_argument("--scope", choices=RECAP_SCOPES, default="full_time")
    args = parser.parse_args(argv)
    asyncio.run(_run(args.file, args.scope))
    return 0


if __name__ == "__main__":
    sys.exit(main())
