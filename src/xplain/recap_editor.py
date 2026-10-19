"""
RecapEditor: one verified recap per scope (full time, half time) in English, Spanish and Greek, from the Foundry model.

Flow: card (recap.recap_card, every number in code) -> facts (recap.recap_facts, categories only) -> model
(structured output RecapOut, with slots) -> recap_verifier.verify_recap (raw checks, fill, filled checks). A rejected
answer is retried (up to MAX_ATTEMPTS calls in all) with the verifier's reasons added to the facts; a timeout is
retried once (TIMEOUT_RETRIES), as for the Narrator; a second timeout, a rejection on the last attempt or any error
gives no text. write() never raises. Each model call is a "recap_editor" llm step in the trace, each check a
"recap_verifier" code step holding that attempt's raw text, and the accepted text a "recap_filler" code step
holding the filled text, with the slots used in its notes. The model, its request parameters and timeout come from models.py, as for the Narrator.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ConfigDict, Field

from .config import FoundryConfig
from .models import DEFAULT_TEMPERATURE, adapt_options, params_for
from .narrator import NarratorAgent
from .recap import RecapCard, RecapOut, recap_facts, slot_values
from .recap_verifier import verify_recap
from .trace import TraceStep, code_step, llm_step

MAX_ATTEMPTS = 3
TIMEOUT_RETRIES = 1


class RecapResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: RecapOut | None      # the filled text; None when no attempt passed the recap verifier
    raw_text: RecapOut | None = None   # the accepted text as the model wrote it, with its slots
    slots: dict[str, str] = {}         # the slots used and the value code put in their place
    attempts: int              # model calls made
    reasons: list[str]         # every rejection, prefixed "attempt N: "
    notes: list[str] = []      # the verifier's notes on the accepted text
    latency_s: float
    trace: list[TraceStep] = []
    # the RecapOut of each attempt the verifier rejected; in memory only, never serialised
    rejected: list[RecapOut] = Field(default_factory=list, exclude=True)

    @property
    def status(self) -> str:
        if self.text is None:
            return "rejected"
        return "accepted" if self.attempts <= 1 else "retried"


def build_recap_agent(credential: Any, config: FoundryConfig | None = None) -> Agent:
    """The Foundry RecapEditor agent (same deployment as the Narrator unless config says otherwise)."""
    from .prompts import recap_instructions

    cfg = config or FoundryConfig.from_env()
    client = OpenAIChatCompletionClient(model=cfg.deployment, base_url=cfg.endpoint, credential=credential)
    return Agent(client, recap_instructions(), name="RecapEditor")


def _texts(out: RecapOut) -> dict[str, str]:
    return {"en": out.text_en, "es": out.text_es, "el": out.text_el}


def _message(facts: dict[str, Any], rejected: list[str]) -> str:
    return json.dumps({**facts, "rejected_because": rejected} if rejected else facts, ensure_ascii=False)


class RecapEditor:
    def __init__(self, agent: NarratorAgent, *, model: str | None = None, timeout_s: float | None = None):
        """agent: anything with the NarratorAgent run() signature (agent_framework.Agent, or a test fake).
        model: the agent's deployment (default FoundryConfig.from_env().deployment), for its parameters."""
        self.agent = agent
        self.model = model or FoundryConfig.from_env().deployment
        self.params = params_for(self.model)
        self.timeout_s = timeout_s if timeout_s is not None else self.params.timeout_s

    async def write(self, card: RecapCard, players: Sequence[str] = ()) -> RecapResult:
        """A verified recap of the card, or none with the reasons. Never raises. players: every player name of the
        match (recap.roster), for the check that no other player is named."""
        start = time.perf_counter()
        reasons: list[str] = []
        trace: list[TraceStep] = []
        candidates: list[RecapOut] = []
        attempts = 0
        options = adapt_options({"response_format": RecapOut, "temperature": DEFAULT_TEMPERATURE}, self.params)

        timeouts = 0

        def result(text: RecapOut | None, notes: list[str] | None = None, raw: RecapOut | None = None,
                   slots: dict[str, str] | None = None) -> RecapResult:
            return RecapResult(text=text, raw_text=raw, slots=slots or {}, attempts=attempts, reasons=reasons,
                               notes=notes or [], latency_s=time.perf_counter() - start, trace=trace,
                               rejected=candidates)

        try:
            facts = recap_facts(card)
            rejected: list[str] = []
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                call_start = time.perf_counter()
                try:
                    response = await asyncio.wait_for(self.agent.run(_message(facts, rejected), options=options),
                                                      timeout=self.timeout_s)
                except Exception as exc:   # recorded here, handled below
                    why = (f"timeout after {self.timeout_s:g} s" if isinstance(exc, TimeoutError)
                           else f"error {type(exc).__name__}: {str(exc)[:200]}")
                    trace.append(llm_step("recap_editor", self.model, attempts, False,
                                          time.perf_counter() - call_start, [why]))
                    if isinstance(exc, TimeoutError) and timeouts < TIMEOUT_RETRIES and attempts < MAX_ATTEMPTS:
                        timeouts += 1
                        reasons.append(f"attempt {attempts}: {why}")
                        continue   # the same message again, as its own attempt
                    raise
                call_s = time.perf_counter() - call_start
                try:
                    out = response.value
                except ValueError as exc:   # pydantic ValidationError is a ValueError
                    out, rejected = None, [f"format: the answer is not a valid RecapOut ({type(exc).__name__})"]
                else:
                    if not isinstance(out, RecapOut):
                        out, rejected = None, ["format: no RecapOut in the answer"]
                if out is None:
                    trace.append(llm_step("recap_editor", self.model, attempts, False, call_s, rejected))
                else:
                    trace.append(llm_step("recap_editor", self.model, attempts, True, call_s))
                    verdict = verify_recap(out, card, players)
                    rejected = verdict.reasons
                    trace.append(code_step("recap_verifier", ok=verdict.ok, attempt=attempts, reasons=verdict.reasons,
                                           notes=verdict.notes, text=_texts(out), checks=verdict.checks))
                    if verdict.ok:
                        values = slot_values(card)
                        slots = {name: values[name] for name in verdict.slots_used}
                        trace.append(code_step("recap_filler", attempt=attempts, text=_texts(verdict.filled),
                                               notes=[f"{{{k}}} = {v}" for k, v in slots.items()]))
                        return result(verdict.filled, verdict.notes, raw=out, slots=slots)
                    candidates.append(out)
                reasons.extend(f"attempt {attempts}: {r}" for r in rejected)
            return result(None)
        except TimeoutError:
            reasons.append(f"attempt {attempts}: timeout after {self.timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001  (write must never raise)
            reasons.append(f"attempt {attempts}: error {type(exc).__name__}: {str(exc)[:200]}")
        return result(None)
