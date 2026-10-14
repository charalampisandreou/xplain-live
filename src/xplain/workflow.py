"""
Workflow: the existing steps as one Agent Framework workflow, Analyst -> Explainer -> Narrator -> Verifier.

    python -m xplain.workflow --file data/match_seed42.jsonl [--max-moments N] [--seed 42]
                              [--mode strict|free|hints]

Each executor is a thin wrapper around code that already exists; none adds rules of its own:
  * AnalystExecutor:   a match's events in; scores every pass and runs the commentary gate
                       (pipeline.gate_match), then sends each commentated moment on, in seq order
  * ExplainerExecutor: one moment in; its SHAP drivers (explainer.ShapExplainer), as a MomentBundle
  * NarratorExecutor:  one bundle in; Narrator.narrate (facts -> model -> Verifier, one retry with the
                       Verifier's reasons, a 15 s timeout per call; it never raises)
  * VerifierExecutor:  the final gate: verifier.verify on the accepted line once more before anything
                       is shown; yields one MomentOutcome per commentated moment

A moment that fails anywhere (no line after the retry, a timeout, an exception in a step) yields a
MomentOutcome with no line and its reasons, and is logged; it never stops the run or the other moments.
Messages to one executor are delivered in order, one at a time, so model calls are sequential.
The minute on a card is Moment.minute_label (clock.match_clock); xP percent is analytics.percent.
"""
import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Never

from agent_framework import Executor, Workflow, WorkflowBuilder, WorkflowContext, handler

from . import analytics
from .config import PHRASE_MODES, AnalyticsConfig, phrase_mode
from .contracts import Moment, MomentBundle, QualifiedBy
from .explainer import ShapExplainer
from .narration import NarratorOut
from .narrator import NarrationResult, Narrator
from .pipeline import ensure_match_file, gate_match, load_events
from .schema import PassEvent
from .scoring import XPassScorer
from .verifier import VerifierResult, verify

log = logging.getLogger(__name__)


# --- messages ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchRequest:
    """The workflow input: a match's events, and at most how many commentated moments to narrate (None: all)."""

    events: Sequence[Any]
    max_moments: int | None = None


@dataclass(frozen=True)
class GatedMoment:
    """Analyst -> Explainer: a commentated Moment, its pass event and why it was commentated."""

    event: PassEvent
    moment: Moment
    qualified_by: tuple[QualifiedBy, ...]


@dataclass(frozen=True)
class Narrated:
    """Narrator -> Verifier: the bundle and the Narrator's result for it."""

    bundle: MomentBundle
    narration: NarrationResult


@dataclass(frozen=True)
class MomentOutcome:
    """The workflow output, one per commentated moment.

    line is the accepted NarratorOut, or None; reasons then says why (Verifier reasons per attempt, a
    timeout, or an error in a step). bundle is None only when the Explainer failed for the moment.
    """

    moment: Moment
    bundle: MomentBundle | None
    line: NarratorOut | None
    verdict: VerifierResult | None   # the final Verifier check of the accepted line; None when there is none
    attempts: int = 0                # model calls made (1 or 2); 0 when the moment never reached the model
    reasons: list[str] = field(default_factory=list)
    latency_s: float = 0.0           # narration wall time

    @property
    def status(self) -> str:
        """"accepted" (first attempt), "retried" (accepted on the retry) or "rejected" (no line)."""
        if self.line is None:
            return "rejected"
        return "accepted" if self.attempts <= 1 else "retried"


def _failed(moment: Moment, step: str, exc: Exception, bundle: MomentBundle | None = None) -> MomentOutcome:
    reason = f"error in {step}: {type(exc).__name__}: {str(exc)[:200]}"
    log.warning("%s %s: %s", moment.moment_id, moment.minute_label, reason)
    return MomentOutcome(moment=moment, bundle=bundle, line=None, verdict=None, reasons=[reason])


# --- executors ---------------------------------------------------------------------------------

class AnalystExecutor(Executor):
    """Scores the match and runs the commentary gate; sends each commentated moment on, in seq order."""

    def __init__(self, scorer: XPassScorer, cfg: AnalyticsConfig = analytics.DEFAULT, id: str = "analyst"):
        super().__init__(id=id)
        self.scorer, self.cfg = scorer, cfg

    @handler
    async def analyse(self, request: MatchRequest, ctx: WorkflowContext[GatedMoment]) -> None:
        live = [GatedMoment(e, m, why) for e, m, why in gate_match(request.events, self.scorer, self.cfg) if why]
        for gated in live[:request.max_moments]:
            await ctx.send_message(gated)


class ExplainerExecutor(Executor):
    """Adds the SHAP drivers of one moment and packs it as a MomentBundle."""

    def __init__(self, scorer: XPassScorer, id: str = "explainer"):
        super().__init__(id=id)
        self.explainer = ShapExplainer(scorer.boosters)

    @handler
    async def explain(self, gated: GatedMoment, ctx: WorkflowContext[MomentBundle, MomentOutcome]) -> None:
        try:
            explanation = self.explainer.explain(gated.event, gated.moment)
            bundle = MomentBundle(moment=gated.moment, explanation=explanation,
                                  qualifies_live=bool(gated.qualified_by), qualified_by=gated.qualified_by)
        except Exception as exc:  # noqa: BLE001  (one moment must not stop the others)
            await ctx.yield_output(_failed(gated.moment, "explainer", exc))
            return
        await ctx.send_message(bundle)


class NarratorExecutor(Executor):
    """Narrator.narrate on one bundle: the model, the Verifier, one retry, the timeout. Never raises."""

    def __init__(self, narrator: Narrator, id: str = "narrator"):
        super().__init__(id=id)
        self.narrator = narrator

    @handler
    async def narrate(self, bundle: MomentBundle, ctx: WorkflowContext[Narrated, MomentOutcome]) -> None:
        try:
            narration = await self.narrator.narrate(bundle)
        except Exception as exc:  # noqa: BLE001  (narrate catches its own errors; this is a last guard)
            await ctx.yield_output(_failed(bundle.moment, "narrator", exc, bundle))
            return
        await ctx.send_message(Narrated(bundle, narration))


class VerifierExecutor(Executor):
    """The final gate: verify() on the accepted line before anything is shown; yields the MomentOutcome."""

    def __init__(self, mode: str | None = None, id: str = "verifier"):
        super().__init__(id=id)
        self.mode = mode or phrase_mode()

    @handler
    async def check(self, item: Narrated, ctx: WorkflowContext[Never, MomentOutcome]) -> None:
        bundle, n = item.bundle, item.narration
        line, verdict, reasons = n.line, None, list(n.reasons)
        try:
            if line is not None:
                verdict = verify(line, bundle, self.mode)
                if not verdict.ok:
                    line = None
                    reasons.extend(f"final check: {r}" for r in verdict.reasons)
        except Exception as exc:  # noqa: BLE001
            await ctx.yield_output(_failed(bundle.moment, "verifier", exc, bundle))
            return
        outcome = MomentOutcome(moment=bundle.moment, bundle=bundle, line=line, verdict=verdict,
                                attempts=n.attempts, reasons=reasons, latency_s=n.latency_s)
        if line is None:
            log.warning("%s %s: no line after %d attempt(s): %s", bundle.moment.moment_id,
                        bundle.moment.minute_label, n.attempts, "; ".join(reasons) or "-")
        await ctx.yield_output(outcome)


# --- building and running ----------------------------------------------------------------------

def build_workflow(scorer: XPassScorer, narrator: Narrator, cfg: AnalyticsConfig = analytics.DEFAULT) -> Workflow:
    """Analyst -> Explainer -> Narrator -> Verifier. Build one per run: a Workflow keeps state between runs.
    The Verifier uses the Narrator's phrase mode."""
    analyst = AnalystExecutor(scorer, cfg)
    explainer = ExplainerExecutor(scorer)
    narrate = NarratorExecutor(narrator)
    verifier = VerifierExecutor(narrator.mode)
    return (WorkflowBuilder(start_executor=analyst, name="xplain-moments")
            .add_edge(analyst, explainer)
            .add_edge(explainer, narrate)
            .add_edge(narrate, verifier)
            .build())


async def run_match(events: Sequence[Any], scorer: XPassScorer, narrator: Narrator,
                    max_moments: int | None = None, cfg: AnalyticsConfig = analytics.DEFAULT) -> list[MomentOutcome]:
    """One MomentOutcome per commentated moment (the first max_moments of them), in seq order."""
    result = await build_workflow(scorer, narrator, cfg).run(MatchRequest(events, max_moments))
    return sorted((o for o in result.get_outputs() if isinstance(o, MomentOutcome)), key=lambda o: o.moment.seq)


# --- CLI ---------------------------------------------------------------------------------------

def outcome_lines(o: MomentOutcome) -> list[str]:
    """Per moment: minute_label, player, xP percent (from code), the three lines and the Verifier verdict."""
    m = o.moment
    rows = [f"{m.minute_label:>5}' | {m.player_name or m.player_id or '-'} | {m.kind} | "
            f"xP {analytics.percent(m.xp)}% | {o.status} ({o.attempts} attempt(s), {o.latency_s:.1f} s)"]
    if o.line is not None:
        rows += [f"  EN: {o.line.line_en}", f"  ES: {o.line.line_es}", f"  EL: {o.line.line_el}",
                 f"  drivers_used: {o.line.drivers_used}"]
        rows += [f"  note: {n}" for n in o.verdict.notes] if o.verdict else []
    else:
        rows += ["  (no line)"] + [f"  reason: {r}" for r in o.reasons]
    return rows


def summary(outcomes: Sequence[MomentOutcome], wall_s: float) -> str:
    """accepted / retried / rejected counts, mean wall seconds per moment and mean narration seconds."""
    n = len(outcomes)
    count = {s: sum(o.status == s for o in outcomes) for s in ("accepted", "retried", "rejected")}
    per = wall_s / n if n else 0.0
    narration = sum(o.latency_s for o in outcomes) / n if n else 0.0
    return (f"{n} moments: accepted {count['accepted']}, retried {count['retried']} (accepted on the retry), "
            f"rejected {count['rejected']}; mean {per:.2f} s per moment ({narration:.2f} s narrating)")


async def _main(path: Path, max_moments: int | None, mode: str) -> None:
    from azure.identity.aio import AzureCliCredential

    from .narrator import build_agent
    from .scoring import find_default_model

    events = load_events(path)
    scorer = XPassScorer(*find_default_model())
    print(f"phrase mode: {mode}")
    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential, mode=mode), mode=mode)
        start = time.perf_counter()
        outcomes = await run_match(events, scorer, narrator, max_moments)
        wall = time.perf_counter() - start
    for o in outcomes:
        print("\n".join(outcome_lines(o)))
    print(summary(outcomes, wall))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Analyst -> Explainer -> Narrator -> Verifier workflow "
                                                 "on a match (needs az login).")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used to generate --file when it does not exist")
    parser.add_argument("--max-moments", type=int, metavar="N", help="narrate only the first N commentated moments")
    parser.add_argument("--mode", choices=PHRASE_MODES, help="phrase mode (default XPLAIN_PHRASE_MODE)")
    args = parser.parse_args(argv)
    if args.max_moments is not None and args.max_moments < 0:
        parser.error("--max-moments must be >= 0")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ensure_match_file(args.file, args.seed)
    asyncio.run(_main(args.file, args.max_moments, args.mode or phrase_mode()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
