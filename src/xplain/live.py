"""
Live spike: one real model call from the running app, with Entra ID auth (never an API key).

GET /api/live/health (api.py) runs LiveCheck.run(): one tiny chat completion ("Reply with OK", at most
MAX_TOKENS output tokens) on the Foundry deployment XPLAIN_LIVE_DEPLOYMENT (default gpt-4.1-mini) at
XPLAIN_FOUNDRY_ENDPOINT (default config.FoundryConfig.endpoint, the /openai/v1/ route). It is off (404) unless
XPLAIN_LIVE=1.

Credential: DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID). In Azure the app's user-assigned
identity (id-xplain-live, Cognitive Services OpenAI User on the Foundry resource) gives the token; locally
`az login` does. The bearer-token provider asks for COGNITIVE_SCOPE on every request; the OpenAI client never sees
a key.

At most one real call per MIN_INTERVAL_S server-wide (successes and failures alike); in between, the last result
comes back with cached=true. A call has TIMEOUT_S in all (token + request). Errors return ok=false with the exception
class and a short message, scrubbed of anything that looks like a token or a key.
"""
from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from .config import FoundryConfig

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_DEPLOYMENT = "gpt-4.1-mini"
PROMPT = "Reply with OK"
MAX_TOKENS = 5
MIN_INTERVAL_S = 60.0
TIMEOUT_S = 15.0
MESSAGE_CHARS = 200

# Anything that could be a credential in an error message: JWTs, "Bearer ...", key=... / api-key: ..., and long
# opaque strings.
_SECRETS = (
    re.compile(r"eyJ[\w-]+\.[\w-]+\.[\w-]*"),
    re.compile(r"(?i)\bbearer\s+\S+"),
    re.compile(r"(?i)\b(api[-_ ]?key|key|token|secret|password|signature|sig)\s*[=:]\s*\S+"),
    re.compile(r"[A-Za-z0-9+/_-]{32,}={0,2}"),
)


def enabled() -> bool:
    """XPLAIN_LIVE=1 turns /api/live/health on."""
    return os.environ.get("XPLAIN_LIVE") == "1"


def deployment() -> str:
    return os.environ.get("XPLAIN_LIVE_DEPLOYMENT") or DEFAULT_DEPLOYMENT


def endpoint() -> str:
    return FoundryConfig.from_env().endpoint


def make_credential() -> Any:
    """The app's managed identity in Azure (AZURE_CLIENT_ID: a user-assigned identity), az login locally."""
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential(managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID"))


def make_client(credential: Any) -> Any:
    """An OpenAI client on the Foundry /openai/v1/ route; the api_key is a bearer-token provider, not a key."""
    from azure.identity import get_bearer_token_provider
    from openai import OpenAI
    return OpenAI(base_url=endpoint(), api_key=get_bearer_token_provider(credential, COGNITIVE_SCOPE),
                  timeout=TIMEOUT_S, max_retries=0)


def credential_name(credential: Any) -> str | None:
    """The credential class that gave the token: DefaultAzureCredential's chosen one, else the class itself."""
    if credential is None:
        return None
    chosen = getattr(credential, "_successful_credential", None)
    return type(chosen if chosen is not None else credential).__name__


def scrub(text: str) -> str:
    """A short error message with anything that looks like a token or a key replaced."""
    for pattern in _SECRETS:
        text = pattern.sub("[redacted]", text)
    text = " ".join(text.split())
    return text[:MESSAGE_CHARS]


class LiveCheck:
    """The rate-limited live call. credential_factory and client_factory are injectable for tests."""

    def __init__(self, credential_factory: Callable[[], Any] = make_credential,
                 client_factory: Callable[[Any], Any] = make_client, clock: Callable[[], float] = time.monotonic,
                 interval_s: float = MIN_INTERVAL_S, timeout_s: float = TIMEOUT_S):
        self._credential_factory, self._client_factory = credential_factory, client_factory
        self._clock, self._interval_s, self._timeout_s = clock, interval_s, timeout_s
        self._lock = threading.Lock()
        self._credential: Any = None
        self._client: Any = None
        self._last: dict | None = None
        self._last_at: float | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")

    def run(self) -> dict:
        """The last result while it is younger than interval_s (cached=true), else one new call."""
        with self._lock:
            now = self._clock()
            if self._last is not None and self._last_at is not None and now - self._last_at < self._interval_s:
                return {**self._last, "cached": True}
            self._last, self._last_at = self._call(), now
            return {**self._last, "cached": False}

    def _call(self) -> dict:
        model = deployment()
        start = time.perf_counter()
        try:
            future = self._pool.submit(self._complete, model)
            reply = future.result(timeout=self._timeout_s)
            return {"ok": True, "model": reply or model, "latency_s": round(time.perf_counter() - start, 3),
                    "credential": credential_name(self._credential)}
        except FutureTimeout:
            return self._error(model, start, "TimeoutError", f"no answer within {self._timeout_s:g} s")
        except Exception as exc:  # noqa: BLE001  (the endpoint reports, never raises)
            return self._error(model, start, type(exc).__name__, str(exc))

    def _complete(self, model: str) -> str | None:
        if self._client is None:
            self._credential = self._credential_factory()
            self._client = self._client_factory(self._credential)
        response = self._client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}], max_completion_tokens=MAX_TOKENS)
        return getattr(response, "model", None)

    def _error(self, model: str, start: float, kind: str, message: str) -> dict:
        return {"ok": False, "model": model, "latency_s": round(time.perf_counter() - start, 3),
                "credential": credential_name(self._credential), "error": kind, "message": scrub(message)}
