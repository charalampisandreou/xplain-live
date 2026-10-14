"""
Narrator: one verified line per Moment in English, Spanish and Greek, from the Foundry model.

Flow per moment: facts -> model (structured output NarratorOut) -> Verifier. A rejected answer is
retried once with the Verifier's reasons added to the facts; a second rejection, a timeout or any
error gives no line. narrate() never raises. Every result is stored by moment_id, so a moment is
narrated at most once and a second call does not reach the model.

The model sees only the facts from build_facts: no numbers, no minute (the card shows those), no SHAP
values, baseline or raw score; difficulty_word comes from narration.DIFFICULTY_TIERS. Drivers whose
label is in phrases.NOT_NARRATED are dropped first, then the lower-ranked driver of any incompatible
pair (phrases.narrated; the rest keep their ranks). Phrase modes (config.phrase_mode): "free", each
driver carries its English label and the model words it naturally; "hints", the label plus suggested
wordings it may use or adapt; "strict", the approved phrases, to be used verbatim. The
Explanation itself is unchanged. Rejected candidates are kept on the NarrationResult in memory only
(excluded from serialisation), for diagnostics. The agent is
injected (build_agent takes any Azure credential, so a managed identity can replace the CLI login).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Protocol

from agent_framework import Agent
from agent_framework.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, ConfigDict, Field

from .config import FoundryConfig, phrase_mode
from .contracts import Driver, MomentBundle
from .narration import NarratorOut, difficulty_tier
from .phrases import driver_phrases, narrated, suggestions
from .prompts import narrator_instructions
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
    # the NarratorOut of each attempt the Verifier rejected; in memory only, never serialised or shown in the UI
    rejected: list[NarratorOut] = Field(default_factory=list, exclude=True)


def build_agent(credential: Any, config: FoundryConfig | None = None, mode: str | None = None) -> Agent:
    """The Foundry Narrator agent. credential: any Azure (async) token credential, e.g. AzureCliCredential.
    mode: the phrase mode whose brief the agent gets (default config.phrase_mode())."""
    cfg = config or FoundryConfig.from_env()
    client = OpenAIChatCompletionClient(model=cfg.deployment, base_url=cfg.endpoint, credential=credential)
    return Agent(client, narrator_instructions(mode), name="Narrator")


def _driver_fact(d: Driver, mode: str) -> dict[str, Any]:
    if mode == "strict":
        return {"rank": d.rank, "topic": d.topic, "direction": d.direction, **driver_phrases(d)}
    fact = {"rank": d.rank, "topic": d.topic, "direction": d.direction, "label": d.label}
    if mode == "hints":
        fact["suggested"] = suggestions(d)
    return fact


# Variation nudge (hints and free modes): the sentence shape, and the outcome wording per kind as an English
# intent the model renders naturally. Neutral verbs only: no cause, no intent or credit.
# "rating_reason": the driver is the reason for the rating ("difficult because it was lofted high, but it still
# reaches a teammate"); "concession": the driver is conceded against the outcome ("is not completed, even
# though it was played along the ground"). Never a reason for the outcome.
SHAPES: tuple[str, ...] = ("rating_reason", "concession")
OUTCOMES: dict[str, tuple[str, ...]] = {
    "surprise_failure": ("goes astray", "is lost", "does not reach a teammate", "comes to nothing"),
    "surprise_success": ("reaches a teammate", "is completed", "comes off"),
}


def variation(bundle: MomentBundle, drivers: list[Driver]) -> dict[str, Any]:
    """A stable nudge for one moment, from a SHA-256 of its moment_id (the same moment always gets the same one).

    drivers are the narrated drivers; only those whose direction matches the tier may be mentioned, so
    situation_facts never exceeds their number, and is 0 when there are none (the brief then asks for the
    difficulty and the outcome only, whatever the shape). "rating_reason" asks for at most one driver (a
    reason clause with two runs over the word limit); "concession" asks for 1 or 2.
    """
    m = bundle.moment
    digest = hashlib.sha256(m.moment_id.encode("utf-8")).digest()
    wanted = "raises" if difficulty_tier(m).family == "easy" else "lowers"
    usable = [d for d in drivers if d.direction == wanted]
    shape = SHAPES[digest[0] % len(SHAPES)]
    facts = 1 if shape == "rating_reason" else 1 + digest[1] % 2
    outcomes = OUTCOMES[m.kind]
    return {"shape": shape, "situation_facts": min(facts, len(usable)), "outcome": outcomes[digest[2] % len(outcomes)]}


def build_facts(bundle: MomentBundle, mode: str | None = None) -> dict[str, Any]:
    """The facts the model may narrate: no numbers, no minute, no SHAP, baseline or raw score.

    Drivers are filtered by phrases.narrated (NOT_NARRATED labels, and the lower-ranked driver of an
    INCOMPATIBLE_PAIRS pair, left out); the others keep their original rank. In "free" mode a driver
    carries its English label exactly as the Explainer emits it; in "hints" mode the label plus
    "suggested" wordings per language (phrases.suggestions); in "strict" mode its approved phrases per
    language (a position driver also its depth and width pieces). The mode defaults to config.phrase_mode().
    Free and hints modes add a "variation" nudge (variation()). The team name is not sent (the card shows it).
    """
    mode = mode or phrase_mode()
    m = bundle.moment
    drivers = narrated(bundle.explanation.drivers)
    facts: dict[str, Any] = {
        "kind": m.kind,
        "player": m.player_name,
        "completed": m.completed,
        "difficulty_word": difficulty_tier(m).words(),
        "drivers": [_driver_fact(d, mode) for d in drivers],
    }
    if mode != "strict":
        facts["variation"] = variation(bundle, drivers)
    return facts


def _message(facts: dict[str, Any], rejected: list[str]) -> str:
    payload = {**facts, "rejected_because": rejected} if rejected else facts
    return json.dumps(payload, ensure_ascii=False)


class Narrator:
    def __init__(self, agent: NarratorAgent, *, timeout_s: float = NARRATOR_TIMEOUT_S, mode: str | None = None):
        """mode: the phrase mode for the facts and the Verifier (default config.phrase_mode()); give the
        agent the same mode's brief (build_agent(..., mode=...))."""
        self.agent = agent
        self.timeout_s = timeout_s
        self.mode = mode or phrase_mode()
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
        candidates: list[NarratorOut] = []   # answers the Verifier rejected, for diagnostics only
        attempts = 0

        def result(line: NarratorOut | None, notes: list[str] | None = None) -> NarrationResult:
            return NarrationResult(line=line, attempts=attempts, reasons=reasons, notes=notes or [],
                                   latency_s=time.perf_counter() - start, rejected=candidates)

        try:
            facts = build_facts(bundle, self.mode)
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
                        checked = verify(out, bundle, self.mode)
                        rejected = checked.reasons
                        if not rejected:
                            return result(out, checked.notes)
                        candidates.append(out)
                reasons.extend(f"attempt {attempts}: {r}" for r in rejected)
            return result(None)
        except TimeoutError:
            reasons.append(f"attempt {attempts}: timeout after {self.timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001  (narrate must never raise)
            reasons.append(f"attempt {attempts}: error {type(exc).__name__}: {str(exc)[:200]}")
        return result(None)
