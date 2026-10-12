import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xplain.narration import NarratorOut
from xplain.narrator import MAX_ATTEMPTS, NarrationResult, Narrator, build_facts
from xplain.prompts import NARRATOR_INSTRUCTIONS
from xplain.verifier import MAX_WORDS

from .test_verifier import make_bundle, make_out

BAD = make_out(numbers_used=["95", "94", "3"], line_en="Weso Wendez misses a 95% pass in minute 3.")  # wrong percent


class FakeAgent:
    """Returns the scripted answers in order: a NarratorOut, an exception to raise, "slow" to hang, or "invalid"."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.messages: list[str] = []

    async def run(self, messages: str, *, options: dict) -> SimpleNamespace:
        assert options["response_format"] is NarratorOut
        self.messages.append(messages)
        answer = self.answers.pop(0)
        if answer == "slow":
            await asyncio.sleep(10)
        if isinstance(answer, Exception):
            raise answer
        if answer == "invalid":
            return InvalidResponse()
        return SimpleNamespace(value=answer)


class InvalidResponse:
    @property
    def value(self):
        return NarratorOut.model_validate_json('{"line_en": "x"}')   # raises ValidationError, like agent_framework


def narrate(agent, bundle=None, **kw) -> NarrationResult:
    return asyncio.run(Narrator(agent, **kw).narrate(bundle or make_bundle()))


# --- accept / retry / reject --------------------------------------------------------------------

def test_good_first_answer_is_accepted():
    agent = FakeAgent(make_out())
    result = narrate(agent)
    assert result.line == make_out()
    assert (result.attempts, result.reasons) == (1, [])
    assert len(agent.messages) == 1
    assert "rejected_because" not in json.loads(agent.messages[0])


def test_bad_then_good_uses_the_retry():
    agent = FakeAgent(BAD, make_out())
    result = narrate(agent)
    assert result.line == make_out()
    assert result.attempts == 2
    assert result.reasons == ["attempt 1: numbers: numbers_used has '95', not in the evidence (allowed ['3', '6', '94'])"]
    retry = json.loads(agent.messages[1])
    assert retry["rejected_because"] == ["numbers: numbers_used has '95', not in the evidence (allowed ['3', '6', '94'])"]
    assert {k: v for k, v in retry.items() if k != "rejected_because"} == build_facts(make_bundle())


def test_bad_twice_gives_no_line():
    agent = FakeAgent(BAD, BAD)
    result = narrate(agent)
    assert result.line is None
    assert result.attempts == MAX_ATTEMPTS == 2
    assert [r.split(":")[0] for r in result.reasons] == ["attempt 1", "attempt 2"]
    assert agent.answers == []


def test_invalid_structured_output_is_retried():
    result = narrate(FakeAgent("invalid", make_out()))
    assert result.line == make_out()
    assert result.reasons == ["attempt 1: format: the answer is not a valid NarratorOut (ValidationError)"]


def test_timeout_gives_no_line():
    result = narrate(FakeAgent("slow"), timeout_s=0.05)
    assert result.line is None
    assert result.reasons == ["attempt 1: timeout after 0.05 s"]
    assert result.latency_s < 5


def test_exception_gives_no_line_and_does_not_raise():
    result = narrate(FakeAgent(RuntimeError("connection reset")))
    assert result.line is None
    assert result.reasons == ["attempt 1: error RuntimeError: connection reset"]


def test_exception_on_retry_gives_no_line():
    result = narrate(FakeAgent(BAD, ConnectionError("gone")))
    assert result.line is None
    assert result.attempts == 2
    assert result.reasons[-1] == "attempt 2: error ConnectionError: gone"


# --- store --------------------------------------------------------------------------------------

def test_second_call_for_the_same_moment_does_not_call_the_model():
    agent = FakeAgent(make_out())
    narrator = Narrator(agent)
    bundle = make_bundle()

    async def twice():
        return await narrator.narrate(bundle), await narrator.narrate(bundle)

    first, second = asyncio.run(twice())
    assert first is second
    assert len(agent.messages) == 1
    assert narrator.stored(bundle.moment.moment_id) is first


def test_concurrent_calls_for_the_same_moment_call_the_model_once():
    agent = FakeAgent(make_out())
    narrator = Narrator(agent)
    bundle = make_bundle()

    async def together():
        return await asyncio.gather(*(narrator.narrate(bundle) for _ in range(3)))

    results = asyncio.run(together())
    assert all(r is results[0] for r in results)
    assert len(agent.messages) == 1


def test_a_rejected_moment_is_stored_too():
    agent = FakeAgent(BAD, BAD)
    narrator = Narrator(agent)
    bundle = make_bundle()
    asyncio.run(narrator.narrate(bundle))
    again = asyncio.run(narrator.narrate(bundle))
    assert again.line is None and len(agent.messages) == 2


def test_result_is_frozen():
    result = narrate(FakeAgent(make_out()))
    with pytest.raises(ValidationError):
        result.attempts = 5


# --- facts --------------------------------------------------------------------------------------

def test_facts_have_no_shap_baseline_or_raw_score():
    facts = build_facts(make_bundle())
    text = json.dumps(facts).lower()
    for word in ("shap", "baseline", "raw_score", "evidence", "delta", "difficulty"):
        assert word not in text
    assert facts == {
        "minute": "3", "minute_added": 0, "kind": "surprise_failure", "player": "Weso Wendez", "team": None,
        "xp_percent": 94, "completed": False,
        "drivers": [{"rank": i, "topic": "pitch position", "label": label, "direction": "raises"}
                    for i, label in enumerate(("along the ground", "from the middle third, out wide",
                                               "with no pressure on the passer"), start=1)],
    }


def test_facts_use_the_minute_label():
    facts = build_facts(make_bundle(minute=45, added=2))
    assert (facts["minute"], facts["minute_added"]) == ("45+2", 2)


def test_prompt_has_the_verifier_word_limits():
    for lang in ("en", "es", "el"):
        assert f"{MAX_WORDS[lang]} in" in NARRATOR_INSTRUCTIONS or f"{MAX_WORDS[lang]} words" in NARRATOR_INSTRUCTIONS
    assert "{" not in NARRATOR_INSTRUCTIONS   # every placeholder was filled
    assert '"with the foot"' in NARRATOR_INSTRUCTIONS


def test_foundry_config_reads_env(monkeypatch):
    from xplain.config import FoundryConfig
    monkeypatch.delenv("XPLAIN_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("XPLAIN_FOUNDRY_DEPLOYMENT", raising=False)
    assert FoundryConfig.from_env() == FoundryConfig()
    assert FoundryConfig().endpoint.endswith("/openai/v1/")
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://other.openai.azure.com/openai/v1/")
    monkeypatch.setenv("XPLAIN_FOUNDRY_DEPLOYMENT", "other-model")
    assert FoundryConfig.from_env() == FoundryConfig("https://other.openai.azure.com/openai/v1/", "other-model")


# --- added time ---------------------------------------------------------------------------------

ADDED_TIME = dict(line_en="In minute 45+2, Weso Wendez misses a 94% pass along the ground.",
                  line_es="En el minuto 45+2, Weso Wendez falla un pase del 94% a ras de suelo.",
                  line_el="Στο 45+2 ο Weso Wendez χάνει μια πάσα 94% κατά μήκος του εδάφους.",
                  drivers_used=[1])


def test_added_time_line_accepted_with_both_digits():
    out = NarratorOut(**ADDED_TIME, numbers_used=["45", "2", "94"])
    result = narrate(FakeAgent(out), make_bundle(minute=45, added=2))
    assert result.line == out and result.attempts == 1


def test_added_time_line_rejected_without_the_added_digit():
    out = NarratorOut(**ADDED_TIME, numbers_used=["45", "94"])
    result = narrate(FakeAgent(out, out), make_bundle(minute=45, added=2))
    assert result.line is None
    assert "attempt 1: numbers: line_en has 2, not in numbers_used" in result.reasons
    assert "attempt 2: numbers: line_en has 2, not in numbers_used" in result.reasons
