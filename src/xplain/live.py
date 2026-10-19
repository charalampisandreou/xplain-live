"""
Live spike: one real model call from the running app, with Entra ID auth (never an API key).

GET /api/live/health (api.py) runs LiveCheck.run(): one tiny chat completion ("Reply with OK", at most
MAX_TOKENS output tokens) on the Foundry deployment XPLAIN_LIVE_DEPLOYMENT (default gpt-4.1-mini) at
XPLAIN_FOUNDRY_ENDPOINT (default config.FoundryConfig.endpoint, the /openai/v1/ route). It is off (404) unless
XPLAIN_LIVE=1.

Credential: DefaultAzureCredential(managed_identity_client_id=AZURE_CLIENT_ID). In Azure the app's user-assigned
identity (id-xplain-live, Cognitive Services OpenAI User on the Foundry resource) gives the token; locally
`az login` does. Two stages: "token" (credential.get_token(COGNITIVE_SCOPE), explicitly, outside the OpenAI call) then
"request" (the chat completion, the token sent as the bearer; the OpenAI client never sees a key). The openai client
reports ANY failure while sending as APIConnectionError "Connection error.", so a token failure must never happen
inside it. "credential" is the class that actually returned this call's token (DefaultAzureCredential's recorded
successful one), null when no token was returned.

At most one real call per MIN_INTERVAL_S server-wide (successes and failures alike), on time.monotonic; in between,
the last result comes back with cached=true and age_s. Both stages share TIMEOUT_S. Errors return ok=false with
"stage", the exception class and a short message, and "causes": up to CAUSE_DEPTH of its __cause__ / __context__
chain, each {"error": class, "message": ...}. Every message is scrubbed of anything that looks like a token or a key.
The scrubbed chain is also logged (warning, logger xplain.live) for the container logs; never a traceback. The body
names the endpoint's host only ("endpoint_host").

Network diagnosis, computed with the call (so at most once per MIN_INTERVAL_S, cached alike): "dns" maps each of
the endpoint host and DNS_HOSTS to socket.getaddrinfo(host, 443) within DNS_TIMEOUT_S ({"ok", "addrs": the first two
IPs, "error": "Class: message" or null}; the hosts are looked up in parallel); "resolv" is the nameserver, search and
options lines of /etc/resolv.conf (null where there is none, e.g. Windows); "proxy" names which of PROXY_VARS are set
(names only, never values). The endpoint host is XPLAIN_FOUNDRY_ENDPOINT's, so it changes without code changes.
"""
from __future__ import annotations

import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any
from urllib.parse import urlparse

from .config import FoundryConfig

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_DEPLOYMENT = "gpt-4.1-mini"
PROMPT = "Reply with OK"
MAX_TOKENS = 5
MIN_INTERVAL_S = 60.0
TIMEOUT_S = 15.0
MESSAGE_CHARS = 200
CAUSE_DEPTH = 3
DNS_TIMEOUT_S = 3.0
DNS_HOSTS = ("plhack-foundry-candreou-resource.cognitiveservices.azure.com", "login.microsoftonline.com",
             "management.azure.com", "pypi.org")
RESOLV_CONF = "/etc/resolv.conf"
RESOLV_KEYS = ("nameserver", "search", "options")
PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")

log = logging.getLogger("xplain.live")

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


def make_client(token: str) -> Any:
    """An OpenAI client on the Foundry /openai/v1/ route. api_key is the Entra ID access token from the token stage
    (sent as "Authorization: Bearer <token>"), never a key."""
    from openai import OpenAI
    return OpenAI(base_url=endpoint(), api_key=token, timeout=TIMEOUT_S, max_retries=0)


def endpoint_host() -> str | None:
    """The endpoint's host only (no path, no query)."""
    return urlparse(endpoint()).hostname


def credential_name(credential: Any) -> str | None:
    """The class of the credential that returned the token: a chain's (DefaultAzureCredential) recorded successful
    one, else the class itself. Called only right after get_token succeeded; a chain that recorded none gives None."""
    if credential is None:
        return None
    chosen = getattr(credential, "_successful_credential", None)
    if chosen is not None:
        return type(chosen).__name__
    return None if hasattr(credential, "credentials") else type(credential).__name__


def scrub(text: str) -> str:
    """A short error message with anything that looks like a token or a key replaced."""
    for pattern in _SECRETS:
        text = pattern.sub("[redacted]", text)
    text = " ".join(text.split())
    return text[:MESSAGE_CHARS]


def error_chain(exc: BaseException) -> list[dict]:
    """The exception and what caused it (__cause__, else __context__), outermost first: {"error": class name,
    "message": scrubbed}; the exception itself plus at most CAUSE_DEPTH causes."""
    out: list[dict] = []
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen and len(out) <= CAUSE_DEPTH:
        seen.add(id(e))
        out.append({"error": type(e).__name__, "message": scrub(str(e))})
        e = e.__cause__ or e.__context__
    return out


def lookup(host: str) -> list[str]:
    """The IPs socket.getaddrinfo gives for host:443, in its order, without repeats."""
    ips: list[str] = []
    for *_, sockaddr in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP):
        if sockaddr[0] not in ips:
            ips.append(sockaddr[0])
    return ips


def dns_report(hosts: list[str], resolver: Callable[[str], list[str]] = lookup,
               timeout_s: float = DNS_TIMEOUT_S) -> dict:
    """{host: {"ok", "addrs" (the first two), "error"}}; all hosts looked up in parallel within timeout_s."""
    hosts = list(dict.fromkeys(h for h in hosts if h))
    out: dict = {}
    if not hosts:
        return out
    pool = ThreadPoolExecutor(max_workers=len(hosts), thread_name_prefix="live-dns")
    futures = {host: pool.submit(resolver, host) for host in hosts}
    deadline = time.perf_counter() + timeout_s
    for host, future in futures.items():
        try:
            addrs = future.result(timeout=max(0.0, deadline - time.perf_counter()))
            out[host] = {"ok": True, "addrs": list(addrs)[:2], "error": None}
        except FutureTimeout:
            out[host] = {"ok": False, "addrs": [], "error": f"TimeoutError: no answer within {timeout_s:g} s"}
        except Exception as exc:  # noqa: BLE001  (a diagnosis reports, never raises)
            out[host] = {"ok": False, "addrs": [], "error": scrub(f"{type(exc).__name__}: {exc}")}
    pool.shutdown(wait=False, cancel_futures=True)   # a stuck lookup never holds the answer
    return out


def resolv_lines(path: str = RESOLV_CONF) -> list[str] | None:
    """The nameserver / search / options lines of resolv.conf; None when the file cannot be read."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    return [scrub(line.strip()) for line in lines if line.split()[:1] and line.split()[0] in RESOLV_KEYS]


def proxy_names(environ: dict | None = None) -> list[str]:
    """Which proxy variables are set (either case), by name only; never their values."""
    env = os.environ if environ is None else environ
    return [name for name in PROXY_VARS if env.get(name) or env.get(name.lower())]


def network_report(resolver: Callable[[str], list[str]] = lookup, resolv_path: str = RESOLV_CONF,
                   environ: dict | None = None, timeout_s: float = DNS_TIMEOUT_S) -> dict:
    """The "dns", "resolv" and "proxy" fields of the body."""
    return {"dns": dns_report([endpoint_host(), *DNS_HOSTS], resolver, timeout_s),
            "resolv": resolv_lines(resolv_path), "proxy": proxy_names(environ)}


class LiveCheck:
    """The rate-limited live call. credential_factory, client_factory and network (the dns / resolv / proxy
    fields) are injectable for tests."""

    def __init__(self, credential_factory: Callable[[], Any] = make_credential,
                 client_factory: Callable[[str], Any] = make_client, clock: Callable[[], float] = time.monotonic,
                 interval_s: float = MIN_INTERVAL_S, timeout_s: float = TIMEOUT_S,
                 network: Callable[[], dict] = network_report):
        self._credential_factory, self._client_factory, self._network = credential_factory, client_factory, network
        self._clock, self._interval_s, self._timeout_s = clock, interval_s, timeout_s
        self._lock = threading.Lock()
        self._credential: Any = None
        self._last: dict | None = None
        self._last_at: float | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live")

    def run(self) -> dict:
        """The last result while it is at most interval_s old on the monotonic clock (cached=true, age_s), else one
        new call."""
        with self._lock:
            now = self._clock()
            if self._last is not None and self._last_at is not None and 0 <= now - self._last_at < self._interval_s:
                return {**self._last, "cached": True, "age_s": round(now - self._last_at, 1)}
            # the diagnosis runs after the call (latency_s is the call's alone) and is cached with it
            self._last, self._last_at = {**self._call(), **self._network()}, now
            return {**self._last, "cached": False, "age_s": 0.0}

    def _call(self) -> dict:
        model = deployment()
        start = time.perf_counter()
        deadline = start + self._timeout_s
        stage, credential = "token", None
        try:
            # the token first, on its own: a token failure must never look like a connection error
            token = self._pool.submit(self._token).result(timeout=self._left(deadline))
            credential = credential_name(self._credential)
            stage = "request"
            reply = self._pool.submit(self._complete, model, token).result(timeout=self._left(deadline))
            return {"ok": True, "model": reply or model, "latency_s": round(time.perf_counter() - start, 3),
                    "credential": credential, "endpoint_host": endpoint_host()}
        except FutureTimeout:
            return self._error(model, start, stage, credential,
                               [{"error": "TimeoutError", "message": f"no answer within {self._timeout_s:g} s"}])
        except Exception as exc:  # noqa: BLE001  (the endpoint reports, never raises)
            chain = error_chain(exc)
            log.warning("live check failed at %s: %s", stage,
                        " <- ".join(f"{c['error']}: {c['message']}" for c in chain))
            return self._error(model, start, stage, credential, chain)

    def _left(self, deadline: float) -> float:
        return max(0.0, deadline - time.perf_counter())

    def _token(self) -> str:
        if self._credential is None:
            self._credential = self._credential_factory()
        return self._credential.get_token(COGNITIVE_SCOPE).token

    def _complete(self, model: str, token: str) -> str | None:
        response = self._client_factory(token).chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}], max_completion_tokens=MAX_TOKENS)
        return getattr(response, "model", None)

    def _error(self, model: str, start: float, stage: str, credential: str | None, chain: list[dict]) -> dict:
        return {"ok": False, "model": model, "latency_s": round(time.perf_counter() - start, 3),
                "credential": credential, "endpoint_host": endpoint_host(), "stage": stage,
                "error": chain[0]["error"], "message": chain[0]["message"], "causes": chain[1:]}
