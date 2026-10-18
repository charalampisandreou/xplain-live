import asyncio
import json
from types import SimpleNamespace

import pytest

from xplain.prompts import RECAP_EXAMPLES
from xplain.recap import RecapOut
from xplain.recap_editor import MAX_ATTEMPTS, RecapEditor, RecapResult

FT, HT, DRAW_FT = RECAP_EXAMPLES
PLAYERS = ["Dario Pell", "Joss Marlin", "Tomas Avery", "Ollie Branck"]
GOOD = FT.out
BAD = GOOD.model_copy(update={"text_en": GOOD.text_en.replace("came from behind", "came from behind 2-1")})


class FakeLLM:
    """No network. Scripted answers in order: a RecapOut, an exception, "slow" to hang, or "invalid"."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.messages: list[dict] = []
        self.options: list[dict] = []

    async def run(self, messages: str, *, options: dict):
        assert options["response_format"] is RecapOut
        self.messages.append(json.loads(messages))
        self.options.append(options)
        answer = self.answers.pop(0)
        if answer == "slow":
            await asyncio.sleep(10)
        if isinstance(answer, Exception):
            raise answer
        if answer == "invalid":
            return Invalid()
        return SimpleNamespace(value=answer)


class Invalid:
    @property
    def value(self):
        return RecapOut.model_validate_json('{"text_en": "x"}')


def write(llm, card=FT.card, **kw) -> RecapResult:
    return asyncio.run(RecapEditor(llm, model="gpt-4.1-mini", **kw).write(card, PLAYERS))


def steps(r: RecapResult) -> list[tuple]:
    return [(s.agent, s.kind, s.attempt, s.ok) for s in r.trace]


def test_good_first_answer():
    llm = FakeLLM(GOOD)
    r = write(llm)
    assert r.raw_text == GOOD and r.status == "accepted" and r.attempts == 1 and r.reasons == []
    assert r.text.text_en == GOOD.text_en.replace("{home_vs_expected}", "15").replace(
        "{away_vs_expected}", "15").replace("{star_name}", "Ollie Branck").replace("{star_pva}", "6")
    assert r.slots == {"home_vs_expected": "15", "away_vs_expected": "15", "star_name": "Ollie Branck",
                       "star_pva": "6"}
    assert steps(r) == [("recap_editor", "llm", 1, True), ("recap_verifier", "code", 1, True),
                        ("recap_filler", "code", 1, True)]
    assert r.trace[1].text == {"en": GOOD.text_en, "es": GOOD.text_es, "el": GOOD.text_el}   # raw
    assert r.trace[2].text["el"] == r.text.text_el and "{star_pva} = 6" in r.trace[2].notes   # filled
    assert r.trace[0].model == "gpt-4.1-mini" and r.trace[0].latency_s is not None
    assert "rejected_because" not in llm.messages[0]


def test_the_model_sees_categories_only():
    llm = FakeLLM(GOOD)
    write(llm)
    text = json.dumps(llm.messages[0], ensure_ascii=False)
    assert not any(ch.isdigit() for ch in text) and "minute" not in text


def test_a_rejected_answer_is_retried_with_the_reasons():
    llm = FakeLLM(BAD, GOOD)
    r = write(llm)
    assert r.raw_text == GOOD and r.status == "retried" and r.attempts == 2
    assert r.reasons and r.reasons[0].startswith("attempt 1: no_numbers")
    assert llm.messages[1]["rejected_because"] == [x.removeprefix("attempt 1: ") for x in r.reasons]
    assert steps(r) == [("recap_editor", "llm", 1, True), ("recap_verifier", "code", 1, False),
                        ("recap_editor", "llm", 2, True), ("recap_verifier", "code", 2, True),
                        ("recap_filler", "code", 2, True)]
    assert r.trace[1].text["en"] == BAD.text_en   # the rejected raw text is in the trace


def test_three_rejections_give_no_text():
    r = write(FakeLLM(BAD, BAD, BAD))
    assert r.text is None and r.status == "rejected" and r.attempts == MAX_ATTEMPTS == 3
    assert [s.ok for s in r.trace if s.agent == "recap_verifier"] == [False, False, False]


def test_timeout_error_and_invalid_answer_never_raise():
    r = write(FakeLLM("slow", "slow"), timeout_s=0.05)
    assert r.text is None and r.reasons == ["attempt 1: timeout after 0.05 s", "attempt 2: timeout after 0.05 s"]
    assert steps(r) == [("recap_editor", "llm", 1, False), ("recap_editor", "llm", 2, False)]
    r = write(FakeLLM("slow", GOOD), timeout_s=0.05)   # one timeout is retried, as for the Narrator
    assert r.raw_text == GOOD and r.attempts == 2 and r.reasons == ["attempt 1: timeout after 0.05 s"]
    r = write(FakeLLM(RuntimeError("gone")))
    assert r.text is None and r.reasons == ["attempt 1: error RuntimeError: gone"]
    r = write(FakeLLM("invalid", GOOD))
    assert r.raw_text == GOOD and r.reasons == ["attempt 1: format: the answer is not a valid RecapOut (ValidationError)"]
    assert steps(r)[0] == ("recap_editor", "llm", 1, False)


@pytest.mark.parametrize("model, timeout, has_temperature", [("gpt-4.1-mini", 15.0, True),
                                                             ("gpt-6.1-sol", 60.0, False)])
def test_model_parameters_come_from_models(model, timeout, has_temperature):
    llm = FakeLLM(GOOD)
    editor = RecapEditor(llm, model=model)
    assert editor.timeout_s == timeout
    asyncio.run(editor.write(FT.card, PLAYERS))
    assert ("temperature" in llm.options[0]) == has_temperature
