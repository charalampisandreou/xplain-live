import asyncio
import logging
from types import SimpleNamespace

import pytest

from xplain.models import DEFAULT_PARAMS, DEFAULT_TIMEOUT_S, MODEL_PARAMS, adapt_options, params_for
from xplain.narration import NarratorOut
from xplain.narrator import TEMPERATURE, Narrator

from .test_verifier import make_bundle, make_out


class RecordingAgent:
    def __init__(self):
        self.options: list[dict] = []

    async def run(self, messages: str, *, options: dict):
        self.options.append(options)
        return SimpleNamespace(value=make_out())


def test_default_model_is_the_baseline():
    assert params_for("gpt-4.1-mini") == DEFAULT_PARAMS
    assert DEFAULT_PARAMS.temperature == TEMPERATURE == 0.2 and DEFAULT_PARAMS.timeout_s == DEFAULT_TIMEOUT_S == 15.0
    assert params_for("not-a-deployment") == DEFAULT_PARAMS and params_for(None) == DEFAULT_PARAMS   # with a warning


def test_reasoning_model_params():
    sol = MODEL_PARAMS["gpt-6.1-sol"]
    assert (sol.temperature, sol.max_completion_tokens, sol.reasoning_effort, sol.timeout_s) == (None, 2000, "low", 60.0)
    base = {"response_format": NarratorOut, "temperature": TEMPERATURE}
    assert adapt_options(base, sol) == {"response_format": NarratorOut, "max_tokens": 2000, "reasoning_effort": "low"}
    assert adapt_options(base, DEFAULT_PARAMS) == base
    assert base == {"response_format": NarratorOut, "temperature": TEMPERATURE}   # not changed in place


@pytest.mark.parametrize("model, timeout, sent", [
    ("gpt-4.1-mini", 15.0, {"response_format": NarratorOut, "temperature": 0.2}),
    ("gpt-6.1-sol", 60.0, {"response_format": NarratorOut, "max_tokens": 2000, "reasoning_effort": "low"}),
])
def test_narrator_takes_the_models_options_and_timeout(model, timeout, sent):
    agent = RecordingAgent()
    narrator = Narrator(agent, mode="hints", model=model)
    assert narrator.timeout_s == timeout
    result = asyncio.run(narrator.narrate(make_bundle()))
    assert result.line is not None and agent.options == [sent]
    assert all(s.model == model for s in result.trace if s.kind == "llm")


def test_model_comes_from_config_and_timeout_can_be_overridden(monkeypatch):
    monkeypatch.setenv("XPLAIN_FOUNDRY_DEPLOYMENT", "gpt-6.1-sol")
    narrator = Narrator(RecordingAgent(), mode="hints")
    assert narrator.model == "gpt-6.1-sol" and narrator.timeout_s == 60.0
    assert Narrator(RecordingAgent(), mode="hints", timeout_s=3.0).timeout_s == 3.0


def test_unknown_model_logs_a_warning_with_the_settings_used(caplog):
    with caplog.at_level(logging.WARNING, logger="xplain.models"):
        assert params_for("gpt-9-turbo") == DEFAULT_PARAMS
    text = caplog.text
    assert "'gpt-9-turbo'" in text and "default settings" in text
    assert "temperature=0.2" in text and "timeout_s=15.0" in text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="xplain.models"):
        params_for("gpt-4.1-mini")
        params_for("gpt-6.1-sol")
    assert caplog.text == ""   # known models: no warning


def test_the_default_deployment_is_gpt_6_1_sol_and_mini_stays_selectable(monkeypatch):
    from xplain.config import FoundryConfig
    monkeypatch.delenv("XPLAIN_FOUNDRY_DEPLOYMENT", raising=False)
    assert FoundryConfig().deployment == FoundryConfig.from_env().deployment == "gpt-6.1-sol"
    narrator = Narrator(RecordingAgent(), mode="hints")
    assert narrator.model == "gpt-6.1-sol" and narrator.timeout_s == 60.0
    monkeypatch.setenv("XPLAIN_FOUNDRY_DEPLOYMENT", "gpt-4.1-mini")
    narrator = Narrator(RecordingAgent(), mode="hints")
    assert narrator.model == "gpt-4.1-mini" and narrator.timeout_s == 15.0
