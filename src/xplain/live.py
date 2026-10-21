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
the last result comes back with cached=true and age_s. Both stages share TIMEOUT_S.

IPv4 only: in the container Azure DNS answers the Foundry names' IPv4 (A) question but always fails their IPv6 (AAAA)
one with EAI_AGAIN (measured 2026-10-19: 8/8 vs 0/8), and glibc then often fails the whole getaddrinfo. So the client
connects through ipv4_http_client(): IPv4Backend looks the host up with AF_INET only and connects to that IP; TLS still
uses the hostname (SNI and the certificate check come from the request URL, not the socket address) and so does the
Host header. The body says "ip_family": "ipv4". The Live retake (retake.py) uses the async twin,
ipv4_async_http_client(): AsyncIPv4Backend retries the IPv4 lookup itself on a gaierror (RETRY_BACKOFF_S) and counts
the retries.

Retry: the request stage is retried ONLY when its exception chain holds a socket.gaierror (name resolution, which is
intermittent for the Foundry hosts in the container): at most len(RETRY_BACKOFF_S) retries, waiting 0.5 s then 1 s,
all inside TIMEOUT_S (a retry whose wait would not leave time to send is not made). Auth errors, HTTP errors and
timeouts are never retried; the token is fetched once and reused. "attempts" (1 to 3, on success and failure) counts
the tries of the stage that ended the call (a token failure is 1). Errors return ok=false with
"stage", the exception class and a short message, and "causes": up to CAUSE_DEPTH of its __cause__ / __context__
chain, each {"error": class, "message": ...}. Every message is scrubbed of anything that looks like a token or a key.
The scrubbed chain is also logged (warning, logger xplain.live) for the container logs; never a traceback. The body
names the endpoint's host only ("endpoint_host").

Network diagnosis, computed with the call (so at most once per MIN_INTERVAL_S, cached alike): "dns" maps each of
the endpoint host and DNS_HOSTS to socket.getaddrinfo(host, 443) within DNS_TIMEOUT_S ({"ok", "addrs": the first two
IPs, "error": "Class: message" or null}, plus "ipv4" and "ipv6": the same three fields for the lookup restricted to
AF_INET / AF_INET6, to tell an IPv6 failure from an IPv4 one; all lookups run in parallel); "resolv" is the nameserver, search and
options lines of /etc/resolv.conf (null where there is none, e.g. Windows); "proxy" names which of PROXY_VARS are set
(names only, never values). The endpoint host is XPLAIN_FOUNDRY_ENDPOINT's, so it changes without code changes.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any
from urllib.parse import urlparse

import httpcore2 as httpcore   # openai 3.x sends through httpx2 / httpcore2 (not httpx / httpcore)
import httpx2 as httpx

from .config import FoundryConfig

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_DEPLOYMENT = "gpt-4.1-mini"
PROMPT = "Reply with OK"
MAX_TOKENS = 5
MIN_INTERVAL_S = 60.0
TIMEOUT_S = 15.0
MESSAGE_CHARS = 200
CAUSE_DEPTH = 3
RETRY_BACKOFF_S = (0.5, 1.0)
IP_FAMILY = "ipv4"
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


def ipv4_address(host: str, port: int) -> str:
    """The first IPv4 address getaddrinfo gives for host:port (an A lookup only, never AAAA)."""
    return socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]


class IPv4Backend(httpcore.SyncBackend):
    """httpcore's network backend, connecting to the host's IPv4 address. Only the TCP connect gets the IP: httpcore
    starts TLS with the request's hostname (SNI, certificate check) and the Host header comes from the URL."""

    def __init__(self, resolve: Callable[[str, int], str] = ipv4_address):
        self._resolve = resolve

    def connect_tcp(self, host: str, port: int, timeout: float | None = None, local_address: str | None = None,
                    socket_options: Iterable[Any] | None = None) -> httpcore.NetworkStream:
        try:
            ip = self._resolve(host, port)
        except OSError as exc:   # a ConnectError like httpcore's own, the gaierror kept as its cause
            raise httpcore.ConnectError(f"{type(exc).__name__}: {exc}") from exc
        return super().connect_tcp(ip, port, timeout=timeout, local_address=local_address,
                                   socket_options=socket_options)


class IPv4Transport(httpx.HTTPTransport):
    """httpx2's transport with IPv4Backend (httpx2 has no public option for a network backend)."""

    def __init__(self, resolve: Callable[[str, int], str] = ipv4_address, **kwargs: Any):
        super().__init__(**kwargs)
        self._pool._network_backend = IPv4Backend(resolve)


def ipv4_http_client(timeout: float = TIMEOUT_S) -> httpx.Client:
    """The http client for Foundry calls from the container: openai's defaults, connecting over IPv4 only."""
    from openai import DefaultHttpxClient
    return DefaultHttpxClient(transport=IPv4Transport(), timeout=timeout)


class AsyncIPv4Backend(httpcore.AnyIOBackend):
    """The async twin of IPv4Backend, for the Agent Framework's AsyncOpenAI client (the Live retake). The IPv4 lookup
    itself is retried on a socket.gaierror (intermittent for the Foundry hosts in the container), waiting backoff_s
    in turn; retries counts them over the backend's life (one backend per retake)."""

    def __init__(self, resolve: Callable[[str, int], str] = ipv4_address,
                 backoff_s: tuple[float, ...] = RETRY_BACKOFF_S,
                 sleep: Callable[[float], Any] = asyncio.sleep):
        self._resolve, self._backoff_s, self._sleep = resolve, backoff_s, sleep
        self.retries = 0

    async def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                          local_address: str | None = None,
                          socket_options: Iterable[Any] | None = None) -> httpcore.AsyncNetworkStream:
        tries = 0
        while True:
            try:
                ip = await asyncio.to_thread(self._resolve, host, port)
                break
            except socket.gaierror as exc:
                if tries >= len(self._backoff_s):
                    raise httpcore.ConnectError(f"{type(exc).__name__}: {exc}") from exc
                wait = self._backoff_s[tries]
                log.warning("live retake: name resolution failed (try %d), retrying in %g s", tries + 1, wait)
                await self._sleep(wait)
                tries += 1
                self.retries += 1
            except OSError as exc:
                raise httpcore.ConnectError(f"{type(exc).__name__}: {exc}") from exc
        return await super().connect_tcp(ip, port, timeout=timeout, local_address=local_address,
                                         socket_options=socket_options)


class AsyncIPv4Transport(httpx.AsyncHTTPTransport):
    """httpx2's async transport with AsyncIPv4Backend."""

    def __init__(self, backend: AsyncIPv4Backend | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.backend = backend or AsyncIPv4Backend()
        self._pool._network_backend = self.backend


def ipv4_async_http_client(backend: AsyncIPv4Backend | None = None,
                           timeout: float = TIMEOUT_S) -> httpx.AsyncClient:
    """The async http client for Foundry calls from the container: openai's defaults, IPv4 only."""
    from openai import DefaultAsyncHttpxClient
    return DefaultAsyncHttpxClient(transport=AsyncIPv4Transport(backend), timeout=timeout)


def make_client(token: str) -> Any:
    """An OpenAI client on the Foundry /openai/v1/ route, over IPv4 only. api_key is the Entra ID access token from
    the token stage (sent as "Authorization: Bearer <token>"), never a key."""
    from openai import OpenAI
    return OpenAI(base_url=endpoint(), api_key=token, timeout=TIMEOUT_S, max_retries=0,
                  http_client=ipv4_http_client(TIMEOUT_S))


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


def is_name_resolution(exc: BaseException) -> bool:
    """True when exc or anything in its __cause__ / __context__ chain is a socket.gaierror."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        if isinstance(e, socket.gaierror):
            return True
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return False


DNS_FAMILIES = (("addrs", socket.AF_UNSPEC), ("ipv4", socket.AF_INET), ("ipv6", socket.AF_INET6))


def lookup(host: str, family: int = socket.AF_UNSPEC) -> list[str]:
    """The IPs socket.getaddrinfo gives for host:443 in family, in its order, without repeats."""
    ips: list[str] = []
    for *_, sockaddr in socket.getaddrinfo(host, 443, family, proto=socket.IPPROTO_TCP):
        if sockaddr[0] not in ips:
            ips.append(sockaddr[0])
    return ips


def dns_report(hosts: list[str], resolver: Callable[[str, int], list[str]] = lookup,
               timeout_s: float = DNS_TIMEOUT_S) -> dict:
    """{host: {"ok", "addrs" (the first two), "error", "ipv4": {...}, "ipv6": {...}}}: the unrestricted lookup, then
    the AF_INET and AF_INET6 ones with the same three fields; all lookups run in parallel within one timeout_s."""
    hosts = list(dict.fromkeys(h for h in hosts if h))
    out: dict = {}
    if not hosts:
        return out
    pool = ThreadPoolExecutor(max_workers=len(hosts) * len(DNS_FAMILIES), thread_name_prefix="live-dns")
    futures = {(host, key): pool.submit(resolver, host, family) for host in hosts for key, family in DNS_FAMILIES}
    deadline = time.perf_counter() + timeout_s
    for host in hosts:
        out[host] = {}
        for key, _ in DNS_FAMILIES:
            result = _answer(futures[host, key], deadline, timeout_s)
            if key == "addrs":
                out[host].update(result)
            else:
                out[host][key] = result
    pool.shutdown(wait=False, cancel_futures=True)   # a stuck lookup never holds the answer
    return out


def _answer(future: Any, deadline: float, timeout_s: float) -> dict:
    """{"ok", "addrs" (the first two), "error"} for one lookup."""
    try:
        addrs = future.result(timeout=max(0.0, deadline - time.perf_counter()))
        return {"ok": True, "addrs": list(addrs)[:2], "error": None}
    except FutureTimeout:
        return {"ok": False, "addrs": [], "error": f"TimeoutError: no answer within {timeout_s:g} s"}
    except Exception as exc:  # noqa: BLE001  (a diagnosis reports, never raises)
        return {"ok": False, "addrs": [], "error": scrub(f"{type(exc).__name__}: {exc}")}


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


def network_report(resolver: Callable[[str, int], list[str]] = lookup, resolv_path: str = RESOLV_CONF,
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
                 network: Callable[[], dict] = network_report, backoff_s: tuple[float, ...] = RETRY_BACKOFF_S,
                 sleep: Callable[[float], None] = time.sleep):
        self._credential_factory, self._client_factory, self._network = credential_factory, client_factory, network
        self._backoff_s, self._sleep = backoff_s, sleep
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
        stage, credential, attempts = "token", None, 1
        try:
            # the token first, on its own: a token failure must never look like a connection error
            token = self._pool.submit(self._token).result(timeout=self._left(deadline))
            credential = credential_name(self._credential)
            stage = "request"
            while True:
                try:
                    reply = self._pool.submit(self._complete, model, token).result(timeout=self._left(deadline))
                    break
                except FutureTimeout:
                    raise
                except Exception as exc:  # noqa: BLE001
                    retries = attempts - 1
                    if not is_name_resolution(exc) or retries >= len(self._backoff_s):
                        raise
                    wait = self._backoff_s[retries]
                    if self._left(deadline) <= wait:   # no time left to send after the wait
                        raise
                    log.warning("live check: name resolution failed (attempt %d), retrying in %g s", attempts, wait)
                    self._sleep(wait)
                    attempts += 1
            return {"ok": True, "model": reply or model, "latency_s": round(time.perf_counter() - start, 3),
                    "credential": credential, "endpoint_host": endpoint_host(), "ip_family": IP_FAMILY,
                    "attempts": attempts}
        except FutureTimeout:
            return self._error(model, start, stage, credential, attempts,
                               [{"error": "TimeoutError", "message": f"no answer within {self._timeout_s:g} s"}])
        except Exception as exc:  # noqa: BLE001  (the endpoint reports, never raises)
            chain = error_chain(exc)
            log.warning("live check failed at %s: %s (attempts %d)", stage,
                        " <- ".join(f"{c['error']}: {c['message']}" for c in chain), attempts)
            return self._error(model, start, stage, credential, attempts, chain)

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

    def _error(self, model: str, start: float, stage: str, credential: str | None, attempts: int,
               chain: list[dict]) -> dict:
        return {"ok": False, "model": model, "latency_s": round(time.perf_counter() - start, 3),
                "credential": credential, "endpoint_host": endpoint_host(), "ip_family": IP_FAMILY, "stage": stage,
                "attempts": attempts,
                "error": chain[0]["error"], "message": chain[0]["message"], "causes": chain[1:]}
