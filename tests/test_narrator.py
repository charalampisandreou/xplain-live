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

BAD = make_out(line_en="Weso Wendez's easy 95% pass goes astray.")  # a number in the line
BAD_REASON = "no_numbers: line_en has '95%'"


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
    assert result.reasons == [f"attempt 1: {BAD_REASON}"]
    retry = json.loads(agent.messages[1])
    assert retry["rejected_because"] == [BAD_REASON]   # the Verifier's reasons are fed back
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

def test_facts_have_no_numbers_minute_shap_baseline_or_raw_score():
    facts = build_facts(make_bundle(minute=45, added=2))
    text = json.dumps(facts, ensure_ascii=False).lower()
    for word in ("shap", "baseline", "raw_score", "evidence", "delta", "minute", "xp", "percent", "45"):
        assert word not in text
    no_ranks = json.dumps({**facts, "drivers": [{k: v for k, v in d.items() if k != "rank"} for d in facts["drivers"]]},
                          ensure_ascii=False)
    assert not any(ch.isdigit() for ch in no_ranks)   # driver ranks are the only numbers the model sees
    assert facts == {
        "kind": "surprise_failure", "player": "Weso Wendez", "team": None, "completed": False,
        "difficulty_word": {"en": "easy", "es": "fácil", "el": "εύκολη"},
        "drivers": [{"rank": i, "topic": "pitch position", "label": label, "direction": "raises"}
                    for i, label in enumerate(("along the ground", "from the middle third, out wide",
                                               "with no pressure on the passer"), start=1)],
    }


def test_difficulty_word_comes_from_the_tier():
    assert build_facts(make_bundle(xp=0.97))["difficulty_word"]["en"] == "very easy"
    assert build_facts(make_bundle(xp=0.15, completed=True))["difficulty_word"] == {
        "en": "very difficult", "es": "muy difícil", "el": "εξαιρετικά δύσκολη"}


def test_prompt_has_the_verifier_word_limits():
    for lang in ("en", "es", "el"):
        assert f"{MAX_WORDS[lang]} in" in NARRATOR_INSTRUCTIONS or f"{MAX_WORDS[lang]} words" in NARRATOR_INSTRUCTIONS
    assert "{" not in NARRATOR_INSTRUCTIONS   # every placeholder was filled
    assert '"with the foot"' in NARRATOR_INSTRUCTIONS
    assert "=== EXAMPLES" in NARRATOR_INSTRUCTIONS and "=== END EXAMPLES ===" in NARRATOR_INSTRUCTIONS
    assert "numbers_used" not in NARRATOR_INSTRUCTIONS


def test_foundry_config_reads_env(monkeypatch):
    from xplain.config import FoundryConfig
    monkeypatch.delenv("XPLAIN_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("XPLAIN_FOUNDRY_DEPLOYMENT", raising=False)
    assert FoundryConfig.from_env() == FoundryConfig()
    assert FoundryConfig().endpoint.endswith("/openai/v1/")
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://other.openai.azure.com/openai/v1/")
    monkeypatch.setenv("XPLAIN_FOUNDRY_DEPLOYMENT", "other-model")
    assert FoundryConfig.from_env() == FoundryConfig("https://other.openai.azure.com/openai/v1/", "other-model")


# --- no numbers and difficulty through the Narrator ---------------------------------------------

def test_minute_word_in_greek_is_retried_with_the_reason():
    bad = make_out(line_el="Στο τελευταίο λεπτό, η εύκολη πάσα του Weso Wendez χάνεται.")
    agent = FakeAgent(bad, make_out())
    result = narrate(agent)
    assert result.line == make_out()
    assert json.loads(agent.messages[1])["rejected_because"] == ["no_numbers: line_el says 'λεπτό'"]


def test_contradicting_difficulty_twice_gives_no_line():
    bad = make_out(line_es="El pase difícil de Weso Wendez se pierde.")
    result = narrate(FakeAgent(bad, bad))
    assert result.line is None
    assert result.reasons == [
        "attempt 1: difficulty: line_es says 'difícil', but the pass is rated easy (easy)",
        "attempt 2: difficulty: line_es says 'difícil', but the pass is rated easy (easy)",
    ]


def test_line_without_difficulty_word_is_accepted_with_a_note():
    plain = make_out(line_en="Weso Wendez's ball along the ground goes astray.")
    result = narrate(FakeAgent(plain))
    assert result.line == plain and result.attempts == 1
    assert result.notes == ["difficulty: line_en has no difficulty word"]
