"""
Narrator: one verified line per Moment in English, Spanish and Greek, from the Foundry model.

Flow per moment: facts -> model (structured output NarratorOut) -> Verifier. A rejected answer is
retried (up to MAX_ATTEMPTS calls in all) with the Verifier's reasons added to the facts. A timeout is retried
once (TIMEOUT_RETRIES), same timeout and same message, as its own attempt within MAX_ATTEMPTS; a second timeout,
a rejection on the last attempt or any error gives no line. narrate() never raises. Every result is stored by moment_id, so a moment is
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
from .models import DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT_S, adapt_options, params_for
from .narration import NarratorOut, difficulty_tier
from .phrases import driver_phrases, narrated, suggestions
from .prompts import narrator_instructions
from .trace import TraceStep, code_step, llm_step
from .verifier import verify

NARRATOR_TIMEOUT_S = DEFAULT_TIMEOUT_S   # per model call, for a deployment not in models.MODEL_PARAMS
MAX_ATTEMPTS = 3            # the first call and up to two retries, each with the Verifier's reasons
TIMEOUT_RETRIES = 1         # a timed-out call is retried this many times (counted in MAX_ATTEMPTS)
TEMPERATURE = DEFAULT_TEMPERATURE


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
    trace: list[TraceStep] = []   # a narrator step per model call and a verifier step per answer (trace.py)
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
    if mode == "hints" and d.feature == "position":
        # the two pieces apart, so the model names one of them and never copies the composed label
        depth_label, width_label = d.label.rsplit(", ", 1)
        hints = suggestions(d)
        return {"rank": d.rank, "topic": d.topic, "direction": d.direction,
                "pieces": {"depth": {"label": depth_label, "suggested": hints["depth"]},
                           "width": {"label": width_label, "suggested": hints["width"]}},
                "use": "one piece only: depth or width, never both"}
    fact = {"rank": d.rank, "topic": d.topic, "direction": d.direction, "label": d.label}
    if mode == "hints":
        fact["suggested"] = suggestions(d)
    return fact


# Variation nudge (hints and free modes): the sentence frame, and the outcome wording per kind as an English
# intent the model renders naturally. Neutral verbs only: no cause, no intent or credit.
# Each frame keeps the honesty contract and differs in where the player, the driver and the result come in the
# sentence (prompts.FRAME_SKETCHES has the sketch the brief shows for each). In the reason frames ("reason",
# "tier_first") the driver is the reason for the rating, next to the difficulty word; in the concession frames
# ("concession", "concession_first", "player_first") it is conceded against the outcome. Never a reason for the
# outcome. FRAME_DRIVERS caps the drivers per frame: a reason clause with two runs over the word limit.
FRAMES: tuple[str, ...] = ("reason", "concession", "concession_first", "player_first", "tier_first")
FRAME_DRIVERS: dict[str, int] = {"reason": 1, "concession": 2, "concession_first": 2, "player_first": 2,
                                 "tier_first": 1}
OUTCOMES: dict[str, tuple[str, ...]] = {
    "surprise_failure": ("does not find a teammate", "is lost", "goes astray", "does not reach a teammate"),
    "surprise_success": ("reaches a teammate", "comes off", "finds a teammate", "is completed"),
}
# The outcome phrase per language (wordings from lines the Verifier has accepted; Spanish "no llega a su destino"
# and the opening "El envío de" by the owner's preference, 2026-10-18); the facts carry all three.
OUTCOME_WORDS: dict[str, dict[str, str]] = {
    "does not find a teammate": {"es": "no encuentra a un compañero", "el": "δεν βρίσκει συμπαίκτη"},
    "is lost": {"es": "se pierde", "el": "χάνεται"},
    "goes astray": {"es": "no llega a su destino", "el": "δεν καταλήγει πουθενά"},
    "does not reach a teammate": {"es": "no llega a un compañero", "el": "δεν φτάνει σε συμπαίκτη"},
    "reaches a teammate": {"es": "llega a un compañero", "el": "φτάνει σε συμπαίκτη"},
    "comes off": {"es": "sale bien", "el": "πετυχαίνει"},
    "finds a teammate": {"es": "encuentra a un compañero", "el": "βρίσκει συμπαίκτη"},
    "is completed": {"es": "se completa", "el": "ολοκληρώνεται"},
}
# Two openings for the frames that start with the pass ("reason", "concession"), alternating by slot.
OPENING_FRAMES: frozenset[str] = frozenset({"reason", "concession"})
OPENINGS: tuple[dict[str, str], ...] = (
    {"en": "{p}'s pass", "es": "El pase de {p}", "el": "Η πάσα του {p}"},
    {"en": "The pass from {p}", "es": "El envío de {p}", "el": "Η μπαλιά του {p}"},
)


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def frame_for(bundle: MomentBundle, slot: int | None = None) -> str:
    """The sentence frame of one moment. slot is the moment's index among the match's commentated moments (the
    Analyst's gate order): frames then rotate through the match from a start fixed by the match_id, so
    neighbouring lines never share a frame and a re-run gives the same frames. Without a slot (a single moment,
    the preview tool) the frame comes from a SHA-256 of the moment_id."""
    m = bundle.moment
    if slot is None:
        return FRAMES[_digest(m.moment_id)[0] % len(FRAMES)]
    return FRAMES[(int.from_bytes(_digest(m.match_id)[:4], "big") + slot) % len(FRAMES)]


def variation(bundle: MomentBundle, drivers: list[Driver], slot: int | None = None) -> dict[str, Any]:
    """A stable nudge for one moment: its frame (frame_for), how many drivers to mention, the outcome phrase per
    language and, for the "reason" and "concession" frames, the opening per language.

    With a slot the outcome rotates through OUTCOMES[kind] by slot (neighbouring moments never share one) and the
    opening alternates by slot (a frame comes back every len(FRAMES) slots, an odd number, so its openings
    alternate too). Without a slot both come from a SHA-256 of the moment_id.

    drivers are the narrated drivers; only those whose direction matches the tier may be mentioned, so
    situation_facts never exceeds their number, and is 0 when there are none (the brief then asks for the
    difficulty and the outcome only, whatever the frame). A frame with FRAME_DRIVERS 1 asks for one driver;
    one with 2 asks for 1 or 2.
    """
    m = bundle.moment
    digest = _digest(m.moment_id)
    wanted = "raises" if difficulty_tier(m).family == "easy" else "lowers"
    usable = [d for d in drivers if d.direction == wanted]
    frame = frame_for(bundle, slot)
    facts = 1 if FRAME_DRIVERS[frame] == 1 else 1 + digest[1] % 2
    outcomes = OUTCOMES[m.kind]
    turn = digest[2] if slot is None else slot
    outcome = outcomes[turn % len(outcomes)]
    nudge: dict[str, Any] = {"frame": frame, "situation_facts": min(facts, len(usable)),
                             "outcome": {"en": outcome, **OUTCOME_WORDS[outcome]}}
    if frame in OPENING_FRAMES:
        opening = OPENINGS[(digest[3] if slot is None else slot) % len(OPENINGS)]
        nudge["opening"] = {lang: text.format(p=m.player_name or "the player") for lang, text in opening.items()}
    return nudge


def build_facts(bundle: MomentBundle, mode: str | None = None, slot: int | None = None) -> dict[str, Any]:
    """The facts the model may narrate: no numbers, no minute, no SHAP, baseline or raw score.

    Drivers are filtered by phrases.narrated (NOT_NARRATED labels, and the lower-ranked driver of an
    INCOMPATIBLE_PAIRS pair, left out); the others keep their original rank. In "free" mode a driver
    carries its English label exactly as the Explainer emits it; in "hints" mode the label plus
    "suggested" wordings per language (phrases.suggestions); in "strict" mode its approved phrases per
    language (a position driver also its depth and width pieces). The mode defaults to config.phrase_mode().
    Free and hints modes add a "variation" nudge (variation(); slot: the moment's index among the match's
    commentated moments, for the frame rotation). The team name is not sent (the card shows it).
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
        facts["variation"] = variation(bundle, drivers, slot)
    return facts


def _message(facts: dict[str, Any], rejected: list[str]) -> str:
    payload = {**facts, "rejected_because": rejected} if rejected else facts
    return json.dumps(payload, ensure_ascii=False)


class Narrator:
    def __init__(self, agent: NarratorAgent, *, timeout_s: float | None = None, mode: str | None = None,
                 model: str | None = None):
        """mode: the phrase mode for the facts and the Verifier (default config.phrase_mode()); give the
        agent the same mode's brief (build_agent(..., mode=...)). model: the agent's deployment (default
        FoundryConfig.from_env().deployment); its request parameters and timeout come from models.MODEL_PARAMS.
        timeout_s overrides the model's timeout."""
        self.agent = agent
        self.model = model or FoundryConfig.from_env().deployment
        self.params = params_for(self.model)
        self.timeout_s = timeout_s if timeout_s is not None else self.params.timeout_s
        self.mode = mode or phrase_mode()
        self._results: dict[str, NarrationResult] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def stored(self, moment_id: str) -> NarrationResult | None:
        return self._results.get(moment_id)

    def remember(self, moment_id: str, result: NarrationResult) -> None:
        """Store an earlier result for this moment, so narrate returns it without calling the model.
        The workflow's final Verifier check still runs on its line."""
        self._results[moment_id] = result

    async def narrate(self, bundle: MomentBundle, slot: int | None = None) -> NarrationResult:
        """The stored result for this moment, or a new one (stored before it is returned). Never raises.
        slot: the moment's index among the match's commentated moments (build_facts), or None."""
        moment_id = bundle.moment.moment_id
        lock = self._locks.setdefault(moment_id, asyncio.Lock())
        async with lock:
            if (done := self._results.get(moment_id)) is not None:
                return done
            result = await self._narrate_once(bundle, slot)
            self._results[moment_id] = result
            return result

    async def _narrate_once(self, bundle: MomentBundle, slot: int | None) -> NarrationResult:
        start = time.perf_counter()
        reasons: list[str] = []
        candidates: list[NarratorOut] = []   # answers the Verifier rejected, for diagnostics only
        trace: list[TraceStep] = []
        attempts = 0
        timeouts = 0
        options = adapt_options({"response_format": NarratorOut, "temperature": TEMPERATURE}, self.params)

        def result(line: NarratorOut | None, notes: list[str] | None = None) -> NarrationResult:
            return NarrationResult(line=line, attempts=attempts, reasons=reasons, notes=notes or [],
                                   latency_s=time.perf_counter() - start, trace=trace, rejected=candidates)

        try:
            facts = build_facts(bundle, self.mode, slot)
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
                    trace.append(llm_step("narrator", self.model, attempts, False,
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
                    out, rejected = None, [f"format: the answer is not a valid NarratorOut ({type(exc).__name__})"]
                    trace.append(llm_step("narrator", self.model, attempts, False, call_s, rejected))
                else:
                    if not isinstance(out, NarratorOut):
                        rejected = ["format: no NarratorOut in the answer"]
                        trace.append(llm_step("narrator", self.model, attempts, False, call_s, rejected))
                    else:
                        trace.append(llm_step("narrator", self.model, attempts, True, call_s))
                        checked = verify(out, bundle, self.mode)
                        rejected = checked.reasons
                        trace.append(code_step("verifier", ok=not rejected, attempt=attempts, reasons=rejected,
                                               notes=checked.notes, checks=checked.checks))
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
