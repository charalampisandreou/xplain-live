"""
Agent trace: one TraceStep per agent per attempt, captured into the match bundle (bundle.Trace) for the Producer
view (the control room: why a line went out, what was checked, which model, how long). Studio never shows it.

Per commentated moment, in order:
  * analyst   (code): the commentary gate let the moment through; notes say how (ranking, threshold)
  * explainer (code): the SHAP drivers were computed (ok False with the error when they were not)
  * narrator  (llm):  one per model call; ok False on a timeout, an error or an answer that is not a NarratorOut
  * verifier  (code): one per answer the Narrator got back, with the Verifier's reasons (rejected) or notes
  * verifier  (code, final=True): the workflow's final check of the accepted line
Latency is recorded for model calls only; code steps carry none. Verifier and recap_verifier steps record the
names of the checks that ran (checks: verifier.CHECKS / recap_verifier.RECAP_CHECKS); None means not recorded (a
bundle built before checks were recorded), never "no checks".

A line reused from an earlier bundle (bundle.stored_lines, also with --no-llm) keeps its Narrator steps marked
reused=True: they ran in an earlier build, not this one, so they carry no latency (attempt is the attempt number
of that earlier build). A reused line with no stored steps gets one reused narrator step saying so. Only the
final check runs again.

A line a producer edited or wrote in the Producer view (line_check.py, never stored in a bundle) has a producer step
(kind "human") and its final Verifier check.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Agent = Literal["analyst", "explainer", "narrator", "verifier", "recap_editor", "recap_verifier", "recap_filler",
                "producer"]
StepKind = Literal["code", "llm", "human"]


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    agent: Agent
    kind: StepKind
    attempt: Annotated[int, Field(ge=1)] = 1
    ok: bool
    reasons: list[str] = []             # why the step failed (Verifier reasons, timeout, error)
    notes: list[str] = []               # what a passing step adds (Verifier notes, how the gate qualified it)
    model: str | None = None            # the deployment, for llm steps only
    latency_s: float | None = None      # wall time, for llm steps only
    final: bool = False                 # the workflow's final Verifier check
    reused: bool = False                # ran in an earlier build (a reused line), not in this one
    text: dict[str, str] | None = None  # recap steps only: the raw text checked (recap_verifier, per attempt) or the
                                        # filled text (recap_filler), by language
    checks: list[str] | None = None     # verifier / recap_verifier steps only: the checks that ran, in order

    @model_validator(mode="after")
    def _check(self) -> TraceStep:
        if self.kind != "llm" and (self.model is not None or self.latency_s is not None):
            raise ValueError("a code or human step has no model and no latency")
        if self.reused and (self.latency_s is not None or self.final):
            raise ValueError("a reused step has no latency and is never the final check")
        if self.checks is not None and self.agent not in ("verifier", "recap_verifier"):
            raise ValueError("only a verifier or recap_verifier step records checks")
        return self


def code_step(agent: Agent, ok: bool = True, attempt: int = 1, reasons: list[str] | None = None,
              notes: list[str] | None = None, final: bool = False, text: dict[str, str] | None = None,
              checks: list[str] | None = None) -> TraceStep:
    return TraceStep(agent=agent, kind="code", attempt=attempt, ok=ok, reasons=reasons or [], notes=notes or [],
                     final=final, text=text, checks=checks)


def reused(steps: list[TraceStep]) -> list[TraceStep]:
    """Steps from an earlier build, marked reused, without latency."""
    return [s.model_copy(update={"reused": True, "latency_s": None}) for s in steps]


def llm_step(agent: Agent, model: str | None, attempt: int, ok: bool, latency_s: float,
             reasons: list[str] | None = None) -> TraceStep:
    return TraceStep(agent=agent, kind="llm", attempt=attempt, ok=ok, reasons=reasons or [], model=model,
                     latency_s=round(latency_s, 3))
