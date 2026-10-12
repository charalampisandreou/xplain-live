"""
Narrator: one verified line per Moment in English, Spanish and Greek, from the Foundry model.

Flow per moment: facts -> model (structured output NarratorOut) -> Verifier. A rejected answer is
retried once with the Verifier's reasons added to the facts; a second rejection, a timeout or any
error gives no line. narrate() never raises. Every result is stored by moment_id, so a moment is
narrated at most once and a second call does not reach the model.

The model sees only the facts from build_facts: no numbers, no minute (the card shows those), no SHAP
values, baseline or raw score; difficulty_word comes from narration.DIFFICULTY_TIERS. The agent is
injected (build_agent takes any Azure credential, so a managed identity can replace the CLI login).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ConfigDict

from .config import FoundryConfig
from .contracts import MomentBundle
from .narration import NarratorOut, difficulty_tier
from .prompts import NARRATOR_INSTRUCTIONS
from .verifier import verify

NARRATOR_TIMEOUT_S = 15.0   # per model call
MAX_ATTEMPTS = 2            # the first call and one retry
TEMPERATURE = 0.2


class NarratorAgent(Protocol):
    """What the Narrator needs from an agent: agent_framework.Agent fits, and so does a test fake."""

    async def run(self, messages: str, *, options: dict[str, Any]) -> Any: ...


class NarrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    line: NarratorOut | None   # None when no attempt passed the Verifier
    attempts: int              # model calls made for this moment (1 or 2)
    reasons: list[str]         # every rejection, prefixed "attempt N: "; empty when accepted first time
    notes: list[str] = []      # the Verifier's notes on the accepted line (e.g. no difficulty word)
    latency_s: float           # wall time of all attempts


def build_agent(credential: Any, config: FoundryConfig | None = None) -> Agent:
    """The Foundry Narrator agent. credential: any Azure (async) token credential, e.g. AzureCliCredential."""
    cfg = config or FoundryConfig.from_env()
    client = OpenAIChatCompletionClient(model=cfg.deployment, base_url=cfg.endpoint, credential=credential)
    return Agent(client, NARRATOR_INSTRUCTIONS, name="Narrator")


def build_facts(bundle: MomentBundle) -> dict[str, Any]:
    """The facts the model may narrate: no numbers, no minute, no SHAP, baseline or raw score."""
    m = bundle.moment
    return {
        "kind": m.kind,
        "player": m.player_name,
        "team": m.team_name,
        "completed": m.completed,
        "difficulty_word": difficulty_tier(m).words(),
        "drivers": [
            {"rank": d.rank, "topic": d.topic, "label": d.label, "direction": d.direction}
            for d in bundle.explanation.drivers
        ],
    }


def _message(facts: dict[str, Any], rejected: list[str]) -> str:
    payload = {**facts, "rejected_because": rejected} if rejected else facts
    return json.dumps(payload, ensure_ascii=False)


class Narrator:
    def __init__(self, agent: NarratorAgent, *, timeout_s: float = NARRATOR_TIMEOUT_S):
        self.agent = agent
        self.timeout_s = timeout_s
        self._results: dict[str, NarrationResult] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def stored(self, moment_id: str) -> NarrationResult | None:
        return self._results.get(moment_id)

    async def narrate(self, bundle: MomentBundle) -> NarrationResult:
        """The stored result for this moment, or a new one (stored before it is returned). Never raises."""
        moment_id = bundle.moment.moment_id
        lock = self._locks.setdefault(moment_id, asyncio.Lock())
        async with lock:
            if (done := self._results.get(moment_id)) is not None:
                return done
            result = await self._narrate_once(bundle)
            self._results[moment_id] = result
            return result

    async def _narrate_once(self, bundle: MomentBundle) -> NarrationResult:
        start = time.perf_counter()
        reasons: list[str] = []
        attempts = 0

        def result(line: NarratorOut | None, notes: list[str] | None = None) -> NarrationResult:
            return NarrationResult(line=line, attempts=attempts, reasons=reasons, notes=notes or [],
                                   latency_s=time.perf_counter() - start)

        try:
            facts = build_facts(bundle)
            rejected: list[str] = []
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                response = await asyncio.wait_for(
                    self.agent.run(_message(facts, rejected),
                                   options={"response_format": NarratorOut, "temperature": TEMPERATURE}),
                    timeout=self.timeout_s,
                )
                try:
                    out = response.value
                except ValueError as exc:   # pydantic ValidationError is a ValueError
                    out, rejected = None, [f"format: the answer is not a valid NarratorOut ({type(exc).__name__})"]
                else:
                    if not isinstance(out, NarratorOut):
                        rejected = ["format: no NarratorOut in the answer"]
                    else:
                        checked = verify(out, bundle)
                        rejected = checked.reasons
                        if not rejected:
                            return result(out, checked.notes)
                reasons.extend(f"attempt {attempts}: {r}" for r in rejected)
            return result(None)
        except TimeoutError:
            reasons.append(f"attempt {attempts}: timeout after {self.timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001  (narrate must never raise)
            reasons.append(f"attempt {attempts}: error {type(exc).__name__}: {str(exc)[:200]}")
        return result(None)
