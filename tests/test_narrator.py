import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xplain.narration import NarratorOut
from xplain.narrator import MAX_ATTEMPTS, NarrationResult, Narrator, build_facts, variation
from xplain.phrases import LABEL_PHRASES as PHRASES
from xplain.phrases import NOT_NARRATED, narrated
from xplain.config import PHRASE_MODES, phrase_mode
from xplain.prompts import INSTRUCTIONS, narrator_instructions
from xplain.verifier import MAX_WORDS, WORD_LIMIT, WORD_TARGET

from .test_verifier import make_bundle, make_driver, make_out

BAD = make_out(line_en="Weso Wendez's easy 95% pass along the ground goes astray.")  # a number in the line
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


def test_bad_three_times_gives_no_line():
    agent = FakeAgent(BAD, BAD, BAD)
    result = narrate(agent)
    assert result.line is None
    assert result.attempts == MAX_ATTEMPTS == 3
    assert [r.split(":")[0] for r in result.reasons] == ["attempt 1", "attempt 2", "attempt 3"]
    assert agent.answers == []


def test_bad_twice_then_good_uses_the_second_retry():
    agent = FakeAgent(BAD, BAD, make_out())
    result = narrate(agent)
    assert result.line == make_out()
    assert result.attempts == 3
    assert result.reasons == [f"attempt 1: {BAD_REASON}", f"attempt 2: {BAD_REASON}"]
    for message in agent.messages[1:]:   # every retry carries the latest Verifier reasons
        assert json.loads(message)["rejected_because"] == [BAD_REASON]


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
    agent = FakeAgent(BAD, BAD, BAD)
    narrator = Narrator(agent)
    bundle = make_bundle()
    asyncio.run(narrator.narrate(bundle))
    again = asyncio.run(narrator.narrate(bundle))
    assert again.line is None and len(agent.messages) == MAX_ATTEMPTS


def test_result_is_frozen():
    result = narrate(FakeAgent(make_out()))
    with pytest.raises(ValidationError):
        result.attempts = 5


# --- facts --------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", PHRASE_MODES)
def test_facts_have_no_numbers_minute_shap_baseline_or_raw_score(mode):
    facts = build_facts(make_bundle(minute=45, added=2), mode)
    text = json.dumps(facts, ensure_ascii=False).lower()
    cleaned = text.replace('"shape"', "")   # the nudge's "shape" key is not "shap"
    for word in ("shap", "baseline", "raw_score", "evidence", "delta", "minute", "xp", "percent", "45"):
        assert word not in cleaned, word
    variation = {k: v for k, v in facts.get("variation", {}).items() if k != "situation_facts"}
    no_ranks = json.dumps({**facts, "drivers": [{k: v for k, v in d.items() if k != "rank"} for d in facts["drivers"]],
                           "variation": variation}, ensure_ascii=False)
    assert not any(ch.isdigit() for ch in no_ranks)   # driver ranks and the nudge's count are the only numbers
    assert "team" not in facts   # the card shows the team; the model does not need it


def test_free_facts_carry_the_english_label():
    bundle = make_bundle(minute=45, added=2)
    assert build_facts(bundle, "free") == {
        "kind": "surprise_failure", "player": "Weso Wendez", "completed": False,
        "difficulty_word": {"en": "easy", "es": "fácil", "el": "εύκολη"},
        "drivers": [   # driver 3 ("with no pressure on the passer") is a default condition and is left out
            {"rank": 1, "topic": "pass height", "direction": "raises", "label": "along the ground"},
            {"rank": 2, "topic": "pitch position", "direction": "raises", "label": "from the middle third, out wide"},
        ],
        "variation": variation(bundle, narrated(bundle.explanation.drivers)),
    }


def test_strict_facts_carry_the_approved_phrases():
    facts = build_facts(make_bundle(minute=45, added=2), "strict")
    assert facts == {   # no variation nudge in strict mode
        "kind": "surprise_failure", "player": "Weso Wendez", "completed": False,
        "difficulty_word": {"en": "easy", "es": "fácil", "el": "εύκολη"},
        "drivers": [   # driver 3 ("with no pressure on the passer") is a default condition and is left out
            {"rank": 1, "topic": "pass height", "direction": "raises",
             "phrase": {"en": "along the ground", "es": "raso", "el": "συρτά"}},
            {"rank": 2, "topic": "pitch position", "direction": "raises",
             "phrase": {"en": "in the middle third, out wide", "es": "en el tercio medio, desde la banda",
                        "el": "στο κέντρο, από τα πλάγια"},
             "depth": {"en": "in the middle third", "es": "en el tercio medio", "el": "στο κέντρο"},
             "width": {"en": "out wide", "es": "desde la banda", "el": "από τα πλάγια"}},
        ],
    }


def test_difficulty_word_comes_from_the_tier():
    assert build_facts(make_bundle(xp=0.97))["difficulty_word"]["en"] == "very easy"
    assert build_facts(make_bundle(xp=0.15, completed=True))["difficulty_word"] == {
        "en": "very difficult", "es": "muy difícil", "el": "εξαιρετικά δύσκολη"}


@pytest.mark.parametrize("mode", PHRASE_MODES)
def test_prompt_has_the_verifier_word_limits(mode):
    text = INSTRUCTIONS[mode]
    assert set(MAX_WORDS.values()) == {WORD_LIMIT}
    assert f"About {WORD_TARGET} words per line, never more than {WORD_LIMIT} words" in text
    assert not re.search(r"\{[a-z_]+\}", text)   # every placeholder was filled
    assert '"with the foot"' in text
    assert "=== EXAMPLES" in text and "=== END EXAMPLES ===" in text
    assert "numbers_used" not in text
    assert "Mention only a driver whose direction matches difficulty_word" in text
    assert '"throw" (say "pass")' in text and "Never give a cause for the outcome" in text


def test_the_examples_block_is_the_same_in_both_modes():
    blocks = {m: INSTRUCTIONS[m][INSTRUCTIONS[m].index("=== EXAMPLES"):] for m in PHRASE_MODES}
    assert blocks["free"] == blocks["strict"]


def test_phrase_mode_defaults_to_hints_and_reads_env(monkeypatch):
    monkeypatch.delenv("XPLAIN_PHRASE_MODE", raising=False)
    assert phrase_mode() == "hints" and narrator_instructions() == INSTRUCTIONS["hints"]
    assert "suggested" in build_facts(make_bundle())["drivers"][0]
    monkeypatch.setenv("XPLAIN_PHRASE_MODE", "Strict")
    assert phrase_mode() == "strict" and narrator_instructions() == INSTRUCTIONS["strict"]
    assert Narrator(FakeAgent()).mode == "strict"
    assert "phrase" in build_facts(make_bundle())["drivers"][0]
    monkeypatch.setenv("XPLAIN_PHRASE_MODE", "loose")
    with pytest.raises(ValueError, match="XPLAIN_PHRASE_MODE"):
        phrase_mode()


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
    bad = make_out(line_el="Στο τελευταίο λεπτό, η εύκολη πάσα του Weso Wendez, συρτά, χάνεται.")
    agent = FakeAgent(bad, make_out())
    result = narrate(agent)
    assert result.line == make_out()
    assert json.loads(agent.messages[1])["rejected_because"] == ["no_numbers: line_el says 'λεπτό'"]


def test_contradicting_difficulty_every_time_gives_no_line():
    bad = make_out(line_es="El pase difícil y raso de Weso Wendez se pierde.")
    result = narrate(FakeAgent(bad, bad, bad))
    assert result.line is None
    assert result.reasons == [
        f"attempt {i}: difficulty: line_es says 'difícil', but the pass is rated easy (easy)" for i in (1, 2, 3)
    ]


def test_line_without_difficulty_word_is_accepted_with_a_note():
    plain = make_out(line_en="Weso Wendez's ball along the ground goes astray.")
    result = narrate(FakeAgent(plain))
    assert result.line == plain and result.attempts == 1
    assert result.notes == ["difficulty: line_en has no difficulty word"]


# --- drivers the model never sees, and phrases --------------------------------------------------

DROPPED = [  # every NOT_NARRATED label
    make_driver(1, "body_part", "body part", "Foot", "with the foot"),
    make_driver(2, "play_pattern", "how the move started", "Regular Play", "in open play"),
    make_driver(3, "under_pressure", "pressure", 0.0, "with no pressure on the passer"),
]
VAGUE = [
    make_driver(1, "body_part", "body part", "Other", "with another part of the body"),
    make_driver(2, "height", "pass height", "High Pass", "lofted high"),
    make_driver(3, "play_pattern", "how the move started", "Other", "from another kind of play"),
]


@pytest.mark.parametrize("mode", PHRASE_MODES)
@pytest.mark.parametrize("drivers", [DROPPED, VAGUE], ids=["defaults", "vague"])
def test_dropped_labels_never_reach_the_facts(drivers, mode):
    text = json.dumps(build_facts(make_bundle(drivers=drivers), mode), ensure_ascii=False).lower()
    for d in (d for d in drivers if d.label in NOT_NARRATED):
        assert d.label not in text
        for lang in ("en", "es", "el"):
            assert PHRASES[d.label][lang].lower() not in text


@pytest.mark.parametrize("mode, driver", [
    ("free", {"rank": 2, "topic": "pass height", "direction": "raises", "label": "lofted high"}),
    ("strict", {"rank": 2, "topic": "pass height", "direction": "raises",
                "phrase": {"en": "lofted high", "es": "por alto", "el": "ψηλά"}}),
])
def test_surviving_drivers_keep_their_rank(mode, driver):
    bundle = make_bundle(drivers=VAGUE)
    assert build_facts(bundle, mode)["drivers"] == [driver]
    assert [d.label for d in bundle.explanation.drivers] == [d.label for d in VAGUE]   # Explanation unchanged


@pytest.mark.parametrize("mode", PHRASE_MODES)
def test_moment_with_every_driver_dropped_still_gets_a_line(mode):
    bundle = make_bundle(drivers=DROPPED)
    assert build_facts(bundle, mode)["drivers"] == []
    line = make_out(line_en="Weso Wendez's easy pass goes astray.", line_es="El pase fácil de Weso Wendez se pierde.",
                    line_el="Η εύκολη πάσα του Weso Wendez χάνεται.", drivers_used=[])
    agent = FakeAgent(line)
    result = narrate(agent, bundle, mode=mode)
    assert result.line == line and result.attempts == 1 and result.reasons == []
    assert json.loads(agent.messages[0])["drivers"] == []


def test_strict_prompt_asks_for_verbatim_phrases_and_at_most_two_drivers():
    text = INSTRUCTIONS["strict"]
    assert "phrase verbatim in the matching language" in text
    assert "more than two drivers" in text
    assert "Mention only the drivers you list in drivers_used" in text
    assert '"without pressure"' in text   # the default conditions, as approved English phrases
    assert 'call the pass "pase"' in text


def test_free_prompt_asks_for_natural_wording_and_guards_the_meaning():
    text = INSTRUCTIONS["free"]
    for rule in ("natural, idiomatic wording in each language", "Do not translate the driver labels word by word",
                 "never midfield", '"out wide" means near the sideline', "is not the penalty box",
                 "Do not give every line the same shape", "Mention at most two situation facts",
                 "adjective before πάσα (συρτή, ψηλή, χαμηλή)",
                 "Mention only the drivers you list in drivers_used"):
        assert rule in text, rule
    for strict_only in ("verbatim", "phrase.en", "depth or width", 'call the pass "pase"'):
        assert strict_only not in text, strict_only
    assert '"with no pressure on the passer"' in text   # the default conditions, as labels


# --- free mode and rejected candidates ----------------------------------------------------------

def test_free_mode_accepts_natural_wording_strict_mode_does_not():
    natural = make_out(line_es="El pase fácil por el suelo de Weso Wendez se pierde.")   # not the phrase "raso"
    assert narrate(FakeAgent(natural), mode="free").line == natural
    strict = narrate(FakeAgent(natural, natural), mode="strict")
    assert strict.line is None
    assert strict.reasons[0] == "attempt 1: phrases: line_es lists driver 1 but does not say 'raso'"


def test_a_rejected_result_keeps_its_candidates_in_memory_only():
    second = make_out(line_en="Weso Wendez's easy 90% pass along the ground goes astray.")
    result = narrate(FakeAgent(BAD, second))
    assert result.line is None and result.rejected == [BAD, second]
    assert "rejected" not in result.model_dump() and "rejected" not in json.loads(result.model_dump_json())


def test_an_accepted_result_has_no_candidates():
    assert narrate(FakeAgent(make_out())).rejected == []


def test_a_retry_keeps_the_rejected_first_candidate():
    result = narrate(FakeAgent(BAD, make_out()))
    assert result.line == make_out() and result.rejected == [BAD]
