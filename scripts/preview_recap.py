"""
Preview one recap with the real RecapEditor model (needs az login). Prints the card, the EN/ES/EL text and the trace;
writes nothing (no bundle).

    python -u scripts/preview_recap.py [--file data/match_seed3001.jsonl] [--scope full_time|half_time]
                                       [--model gpt-6.1-sol]

--model picks the Foundry deployment (default XPLAIN_FOUNDRY_DEPLOYMENT); its parameters come from
xplain.models. Every call's tokens are printed, with the cost at compare_models.PRICES (output tokens include
any reasoning tokens).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from xplain.recap import RECAP_SCOPES, RecapCard
from xplain.recap_editor import RecapResult


def usage_lines(calls: list, price: tuple[float, float] | None) -> list[str]:
    """Tokens per call (compare_models.CallRecord) and the cost at price (USD per 1M input, output tokens)."""
    tok_in = sum(c.input_tokens or 0 for c in calls)
    tok_out = sum(c.output_tokens or 0 for c in calls)
    reasoning = sum(c.reasoning_tokens or 0 for c in calls)
    cost = "n/a" if price is None else f"{(tok_in * price[0] + tok_out * price[1]) / 1e6:.4f} USD"
    return [f"tokens: {len(calls)} call(s), input {tok_in}, output {tok_out} (of which reasoning {reasoning}); "
            f"cost {cost}"]


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
    if r.raw_text is not None:
        rows += ["  raw (as the model wrote it):", f"    EN: {r.raw_text.text_en}", f"    ES: {r.raw_text.text_es}",
                 f"    EL: {r.raw_text.text_el}"]
    if r.text is not None:
        rows += ["  filled (shown):", f"    EN: {r.text.text_en}", f"    ES: {r.text.text_es}",
                 f"    EL: {r.text.text_el}", f"  claims_used: {r.text.claims_used}", f"  slots: {r.slots}"]
    rows += [f"  reason: {x}" for x in r.reasons] + [f"  note: {x}" for x in r.notes]
    for i, c in enumerate(r.rejected, start=1):
        rows += [f"  rejected candidate {i}:", f"    EN: {c.text_en}", f"    ES: {c.text_es}", f"    EL: {c.text_el}",
                 f"    claims_used: {c.claims_used}"]
    rows += ["trace:"] + [f"  {s.agent} ({s.kind}) attempt {s.attempt}: {'pass' if s.ok else 'fail'}"
                          + (f", {s.model}, {s.latency_s:.2f} s" if s.kind == "llm" else "")
                          + (f" | {'; '.join(s.reasons)}" if s.reasons else "") for s in r.trace]
    return rows


async def _run(path: Path, scope: str, model: str | None) -> None:
    from azure.identity.aio import AzureCliCredential
    from compare_models import PRICES, ModelAgent

    from xplain.bundle import _load_scorer
    from xplain.config import FoundryConfig
    from xplain.models import params_for
    from xplain.pipeline import load_events, pass_marks
    from xplain.recap import recap_card, roster
    from xplain.recap_editor import RecapEditor, build_recap_agent

    events = load_events(path)
    card = recap_card(events, pass_marks(events, _load_scorer()), scope)
    print("\n".join(card_lines(card)), flush=True)
    cfg = FoundryConfig.from_env() if model is None else FoundryConfig(deployment=model)
    async with AzureCliCredential() as credential:
        agent = ModelAgent(build_recap_agent(credential, cfg), params_for(cfg.deployment))
        editor = RecapEditor(agent, model=cfg.deployment)
        print(f"model: {editor.model}", flush=True)
        result = await editor.write(card, roster(events))
    print("\n".join(result_lines(result) + usage_lines(agent.calls, PRICES.get(cfg.deployment))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one recap with the real model and print it (no bundle).")
    parser.add_argument("--file", type=Path, default=Path("data/match_seed3001.jsonl"))
    parser.add_argument("--scope", choices=RECAP_SCOPES, default="full_time")
    parser.add_argument("--model", help="Foundry deployment (default XPLAIN_FOUNDRY_DEPLOYMENT)")
    args = parser.parse_args(argv)
    asyncio.run(_run(args.file, args.scope, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
