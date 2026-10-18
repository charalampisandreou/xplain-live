"""
Agent trace: one TraceStep per agent per attempt, captured into the match bundle (bundle.Trace) for a later
Analyst screen. Studio never shows it.

Per commentated moment, in order:
  * analyst   (code): the commentary gate let the moment through; notes say how (ranking, threshold)
  * explainer (code): the SHAP drivers were computed (ok False with the error when they were not)
  * narrator  (llm):  one per model call; ok False on a timeout, an error or an answer that is not a NarratorOut
  * verifier  (code): one per answer the Narrator got back, with the Verifier's reasons (rejected) or notes
  * verifier  (code, final=True): the workflow's final check of the accepted line
Latency is recorded for model calls only; code steps carry none.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Agent = Literal["analyst", "explainer", "narrator", "verifier", "recap_editor", "recap_verifier"]
StepKind = Literal["code", "llm"]


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


def code_step(agent: Agent, ok: bool = True, attempt: int = 1, reasons: list[str] | None = None,
              notes: list[str] | None = None, final: bool = False) -> TraceStep:
    return TraceStep(agent=agent, kind="code", attempt=attempt, ok=ok, reasons=reasons or [], notes=notes or [],
                     final=final)


def llm_step(agent: Agent, model: str | None, attempt: int, ok: bool, latency_s: float,
             reasons: list[str] | None = None) -> TraceStep:
    return TraceStep(agent=agent, kind="llm", attempt=attempt, ok=ok, reasons=reasons or [], model=model,
                     latency_s=round(latency_s, 3))
