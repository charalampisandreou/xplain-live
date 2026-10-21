"""
Live retake: one commentated moment of a served match bundle narrated again, live, by the Foundry Narrator and
checked by the Verifier, from the running app (the Producer's "Live retake" button).

POST /api/live/retake (api.py), body {"match": name, "moment_id": id}; 404 unless XPLAIN_LIVE=1, like
/api/live/health. The moment's facts come from the bundle as stored (MomentBundle: the moment and its SHAP drivers),
so the Analyst and the Explainer are not run again and no number is computed again. Its slot is its index among the
bundle's commentated moments in seq order, as in the build, so the Narrator gets the same frame. Then, as in the
workflow: narrator.Narrator (up to narrator.MAX_ATTEMPTS calls, each answer checked by the Verifier, a rejected one
retried with the reasons) and the final Verifier check (final_step, the twin of workflow.final_step: workflow.py is not
imported, it would pull the build pipeline into the container) of the accepted line, in the bundle's
phrase mode. Nothing is stored: the bundle and what Studio puts on air stay as they are.

Model: XPLAIN_LIVE_DEPLOYMENT (live.deployment(), default gpt-4.1-mini), at XPLAIN_FOUNDRY_ENDPOINT. Auth as in
live.py: the managed identity (async DefaultAzureCredential, AZURE_CLIENT_ID) in Azure, az login locally; the token is
fetched first, on its own ("token" stage), and sent as the bearer (never a key). The client connects over IPv4 only
(live.ipv4_async_http_client); its lookup retries are reported as network_retries.

Guards, server-wide: one retake at a time (429 "busy"), COOLDOWN_S between the end of one and the start of the next
(429 "cooldown" with retry_after_s), at most DAILY_CAP a UTC day (429 "cap"), failures included. Errors are scrubbed
(live.scrub / live.error_chain), never a traceback.

The answer: status accepted (first attempt) | retried (accepted on a later attempt) | rejected (no attempt passed the
Verifier) | failed (an error: token, timeout, Foundry, network), line {en, es, el} or null, attempts, reasons, notes,
model, latency_s (server wall time of the whole retake), network_retries, steps (trace.TraceStep dicts: a narrator
step per call, a verifier step per answer, the final verifier step), and on a failure "stage", "error", "message".
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import live
from .config import phrase_mode
from .contracts import MomentBundle
from .narrator import Narrator, NarratorAgent
from .trace import TraceStep, code_step
from .verifier import VerifierResult, verify

COOLDOWN_S = 10.0
DAILY_CAP = 200


class RetakeRefused(Exception):
    """A retake that is not run: status is the HTTP status, reason a short word, detail the message."""

    def __init__(self, status: int, reason: str, detail: str, retry_after_s: float | None = None):
        super().__init__(detail)
        self.status, self.reason, self.detail, self.retry_after_s = status, reason, detail, retry_after_s

    def body(self) -> dict:
        out: dict[str, Any] = {"ok": False, "refused": self.reason, "detail": self.detail}
        if self.retry_after_s is not None:
            out["retry_after_s"] = self.retry_after_s
        return out


def find_moment(bundle: dict, moment_id: str) -> tuple[MomentBundle, int]:
    """The commentated moment's MomentBundle and its slot (index among the commentated moments by seq)."""
    live_moments = sorted((m["bundle"] for m in bundle.get("moments", []) if m["bundle"].get("qualifies_live")),
                          key=lambda b: b["moment"]["seq"])
    for slot, b in enumerate(live_moments):
        if b["moment"]["moment_id"] == moment_id:
            return MomentBundle.model_validate(b), slot
    raise RetakeRefused(404, "moment", f"no commentated moment {moment_id!r} in this match")


def make_credential() -> Any:
    """The async twin of live.make_credential: the managed identity in Azure, az login locally."""
    from azure.identity.aio import DefaultAzureCredential
    return DefaultAzureCredential(managed_identity_client_id=os.environ.get("AZURE_CLIENT_ID"))


def make_agent(token: str, model: str, mode: str, backend: live.AsyncIPv4Backend) -> NarratorAgent:
    """The Narrator agent on an AsyncOpenAI client at the Foundry /openai/v1/ route, over IPv4 only; api_key is the
    Entra ID access token (sent as the bearer), never a key."""
    from agent_framework import Agent
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    from .prompts import narrator_instructions
    client = AsyncOpenAI(base_url=live.endpoint(), api_key=token, max_retries=0,
                         http_client=live.ipv4_async_http_client(backend))
    return Agent(OpenAIChatCompletionClient(model=model, async_client=client), narrator_instructions(mode),
                 name="Narrator")


class Retake:
    """The guarded retake. credential_factory, agent_factory, the clocks and the bundle loader are injectable for
    tests."""

    def __init__(self, credential_factory: Callable[[], Any] = make_credential,
                 agent_factory: Callable[[str, str, str, live.AsyncIPv4Backend], NarratorAgent] = make_agent,
                 clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], dt.date] = lambda: dt.datetime.now(dt.UTC).date(),
                 cooldown_s: float = COOLDOWN_S, cap: int = DAILY_CAP):
        self._credential_factory, self._agent_factory = credential_factory, agent_factory
        self._clock, self._today, self._cooldown_s, self._cap = clock, today, cooldown_s, cap
        self._credential: Any = None
        self._lock = asyncio.Lock()
        self._last_end: float | None = None
        self._day: dt.date | None = None
        self._count = 0
        self._bundles: dict[Path, dict] = {}

    def _bundle(self, path: Path) -> dict:
        if path not in self._bundles:
            if not path.is_file():
                raise RetakeRefused(404, "match", "no such match")
            with path.open(encoding="utf-8") as f:
                self._bundles[path] = json.load(f)
        return self._bundles[path]

    def _admit(self) -> None:
        """Refuses a retake over the guards; counts an admitted one."""
        if self._lock.locked():
            raise RetakeRefused(429, "busy", "a retake is already running")
        now = self._clock()
        if self._last_end is not None and now - self._last_end < self._cooldown_s:
            wait = round(self._cooldown_s - (now - self._last_end), 1)
            raise RetakeRefused(429, "cooldown", f"ready again in {wait:g} s", retry_after_s=wait)
        if self._day != (day := self._today()):
            self._day, self._count = day, 0
        if self._count >= self._cap:
            raise RetakeRefused(429, "cap", f"the day's {self._cap} retakes are used up")
        self._count += 1

    async def run(self, bundle_path: Path, moment_id: str) -> dict:
        bundle = self._bundle(bundle_path)
        moment, slot = find_moment(bundle, moment_id)
        self._admit()
        async with self._lock:
            try:
                return await self._run(bundle, moment, slot)
            finally:
                self._last_end = self._clock()

    async def _token(self) -> str:
        if self._credential is None:
            self._credential = self._credential_factory()
        return (await self._credential.get_token(live.COGNITIVE_SCOPE)).token

    async def _run(self, bundle: dict, moment: MomentBundle, slot: int) -> dict:
        model = live.deployment()
        mode = bundle.get("meta", {}).get("phrase_mode") or phrase_mode()
        backend = live.AsyncIPv4Backend()
        start = time.perf_counter()
        base = {"moment_id": moment.moment.moment_id, "model": model, "mode": mode}
        try:
            token = await asyncio.wait_for(self._token(), timeout=live.TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001  (the endpoint reports, never raises)
            chain = ([{"error": "TimeoutError", "message": f"no token within {live.TIMEOUT_S:g} s"}]
                     if isinstance(exc, TimeoutError) else live.error_chain(exc))
            live.log.warning("live retake failed at token: %s",
                             " <- ".join(f"{c['error']}: {c['message']}" for c in chain))
            return {**base, "ok": False, "status": "failed", "stage": "token", "error": chain[0]["error"],
                    "message": chain[0]["message"], "line": None, "attempts": 0, "reasons": [], "notes": [],
                    "steps": [], "network_retries": 0, "latency_s": round(time.perf_counter() - start, 3)}
        narrator = Narrator(self._agent_factory(token, model, mode, backend), mode=mode, model=model)
        result = await narrator.narrate(moment, slot)
        steps: list[TraceStep] = list(result.trace)
        line, reasons, notes = result.line, [live.scrub(r) for r in result.reasons], list(result.notes)
        if line is not None:
            verdict = verify(line, moment, mode)
            steps.append(final_step(verdict, result.attempts))
            notes = list(verdict.notes)
            if not verdict.ok:
                line = None
                reasons += [f"final check: {r}" for r in verdict.reasons]
        if line is not None:
            status = "accepted" if result.attempts <= 1 else "retried"
        else:   # an error or a timeout ends on a failed narrator step; a rejection on a verifier step
            status = "failed" if steps and steps[-1].agent == "narrator" and not steps[-1].ok else "rejected"
        out = {**base, "ok": line is not None, "status": status,
               "line": None if line is None else {"en": line.line_en, "es": line.line_es, "el": line.line_el},
               "attempts": result.attempts, "reasons": reasons, "notes": notes,
               "steps": [_scrubbed(s) for s in steps], "network_retries": backend.retries,
               "latency_s": round(time.perf_counter() - start, 3)}
        if status == "failed":
            out.update(stage="request", error="narrator", message=reasons[-1] if reasons else "")
        return out


def _scrubbed(step: TraceStep) -> dict:
    d = step.model_dump(mode="json")
    d["reasons"] = [live.scrub(r) for r in d["reasons"]]
    return d


def final_step(verdict: VerifierResult, attempt: int) -> TraceStep:
    """The trace step of the final Verifier check, as workflow.final_step makes it (a test keeps them equal)."""
    return code_step("verifier", ok=verdict.ok, attempt=max(attempt, 1), reasons=list(verdict.reasons),
                     notes=list(verdict.notes), final=True, checks=list(verdict.checks))
