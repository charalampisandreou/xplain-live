"""
Per-model request parameters for the Foundry deployments the Narrator (and later the RecapEditor) can use.

The deployment is chosen by config only (XPLAIN_FOUNDRY_DEPLOYMENT, config.FoundryConfig); this table says how
to call it. The values come from scripts/compare_models.py, where each was measured:
  * gpt-4.1-mini: the Narrator model today, temperature 0.2, 15 s per call
  * gpt-4.1:      same parameters (tried and dropped: near-identical lines, about 5x the cost)
  * gpt-6.1-sol:  a reasoning model that rejects any temperature but the default 1, so none is sent; it takes
                  reasoning_effort and max_completion_tokens, and gets 60 s per call
A deployment not in the table gets DEFAULT_PARAMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT_S = 15.0    # per model call


@dataclass(frozen=True)
class ModelParams:
    """The request parameters of one deployment. None: the parameter is not sent."""

    temperature: float | None = DEFAULT_TEMPERATURE
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S


DEFAULT_PARAMS = ModelParams()
MODEL_PARAMS: dict[str, ModelParams] = {
    "gpt-4.1-mini": DEFAULT_PARAMS,
    "gpt-4.1": DEFAULT_PARAMS,
    # rejects temperature != 1 (so none is sent: the default 1); a reasoning model, hence the larger budget
    "gpt-6.1-sol": ModelParams(temperature=None, max_completion_tokens=2000, reasoning_effort="low", timeout_s=60.0),
}


def params_for(model: str | None) -> ModelParams:
    """The parameters of a deployment; DEFAULT_PARAMS for one not in MODEL_PARAMS (or None)."""
    return MODEL_PARAMS.get(model or "", DEFAULT_PARAMS)


def adapt_options(options: dict[str, Any], params: ModelParams) -> dict[str, Any]:
    """Request options for one deployment: its temperature (or none) and its extra parameters.
    The given dict is not changed."""
    out = {k: v for k, v in options.items() if k != "temperature"}
    if params.temperature is not None:
        out["temperature"] = params.temperature
    if params.max_completion_tokens is not None:
        out["max_tokens"] = params.max_completion_tokens   # the chat client sends it as max_completion_tokens
    if params.reasoning_effort is not None:
        out["reasoning_effort"] = params.reasoning_effort
    return out
