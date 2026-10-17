import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_models as cm  # noqa: E402

from xplain.narration import NarratorOut  # noqa: E402
from xplain.narrator import TEMPERATURE, Narrator  # noqa: E402

from .test_line_variety import _entry  # noqa: E402
from .test_verifier import make_bundle, make_out  # noqa: E402

BAD = make_out(line_en="Weso Wendez's easy 95% pass goes astray, even though it was along the ground.")  # a number


class FakeAgent:
    """Scripted answers in order, each with token usage; records the options it was sent."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.options: list[dict] = []

    async def run(self, messages: str, *, options: dict) -> SimpleNamespace:
        self.options.append(options)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(value=answer, usage_details={"input_token_count": 1000, "output_token_count": 50,
                                                            "reasoning_output_token_count": 20})


def test_options_per_model():
    base = {"response_format": NarratorOut, "temperature": TEMPERATURE}
    assert cm.adapt_options(base, cm.MODEL_PARAMS["gpt-4.1-mini"]) == base
    assert cm.adapt_options(base, cm.MODEL_PARAMS["gpt-4.1"]) == base
    sol = cm.adapt_options(base, cm.MODEL_PARAMS["gpt-6.1-sol"])
    assert sol == {"response_format": NarratorOut, "max_tokens": 2000, "reasoning_effort": "low"}
    assert "temperature" not in sol and cm.MODEL_PARAMS["gpt-6.1-sol"].timeout_s == 60.0
    assert base == {"response_format": NarratorOut, "temperature": TEMPERATURE}   # the Narrator's dict is not changed


def test_model_agent_records_calls_through_the_real_narrator():
    fake = FakeAgent(BAD, make_out())
    agent = cm.ModelAgent(fake, cm.MODEL_PARAMS["gpt-6.1-sol"])
    result = asyncio.run(Narrator(agent, mode="hints").narrate(make_bundle()))
    assert result.line == make_out() and result.attempts == 2   # the Verifier rejected the first, as usual
    assert all("temperature" not in o and o["reasoning_effort"] == "low" for o in fake.options)
    assert [(c.input_tokens, c.output_tokens, c.reasoning_tokens, c.error) for c in agent.calls] == [
        (1000, 50, 20, None)] * 2


def test_model_agent_records_an_error():
    agent = cm.ModelAgent(FakeAgent(RuntimeError("boom")), cm.MODEL_PARAMS["gpt-4.1"])
    result = asyncio.run(Narrator(agent, mode="hints").narrate(make_bundle()))
    assert result.line is None and "RuntimeError" in result.reasons[0]
    assert agent.calls[0].error == "RuntimeError"


def test_percentile():
    assert cm.percentile([], 50) is None
    assert cm.percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert cm.percentile([float(i) for i in range(1, 21)], 95) == 19.0


def test_report_from_saved_runs(tmp_path):
    line = {"line_en": "Even though it was lofted high, the pass reaches a teammate.", "line_es": "El pase llega.",
            "line_el": "Η πάσα φτάνει.", "drivers_used": []}
    for name, status in (("gpt-4.1-mini", "accepted"), ("gpt-6.1-sol", "retried")):
        e1, e2 = _entry(10, True, line, status), _entry(20, True, None, "rejected")
        e1["narration"].update(latency_s=1.5, reasons=["attempt 1: no_numbers: x"] if status == "retried" else [])
        e2["narration"].update(latency_s=4.0, reasons=["attempt 3: words: too long"])
        cm.bundle_path(name, tmp_path).write_text(json.dumps({"moments": [e1, e2]}), encoding="utf-8")
        cm.calls_path(name, tmp_path).write_text(json.dumps(
            {"model": name, "params": {"temperature": 0.2, "reasoning_effort": None}, "wall_s": 5.0,
             "calls": [{"latency_s": 1.0, "input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": None,
                        "error": None}] * 4}), encoding="utf-8")
    text = cm.write_report(["gpt-4.1-mini", "gpt-6.1-sol", "gpt-4.1"], tmp_path).read_text(encoding="utf-8")
    assert "| gpt-4.1-mini | temperature=0.2 | 2 | 1 | 0 | 1 | 4 (0) |" in text
    assert "| gpt-6.1-sol | temperature=0.2 | 2 | 0 | 1 | 1 |" in text
    assert "| 0.0022 |" in text and "| n/a |" in text   # 4000 in, 400 out at mini prices; no price for sol
    assert "attempt 1: no_numbers: x" in text and "attempt 3: words: too long" in text
    assert text.count("- **gpt-4.1-mini**: Even though it was lofted high") == 1
    assert "- **gpt-6.1-sol**: (no line: rejected)" in text
    assert "- **gpt-4.1-mini EL**: Η πάσα φτάνει." in text
    assert "gpt-4.1 |" not in text.replace("gpt-4.1-mini", "")   # no saved run: skipped
