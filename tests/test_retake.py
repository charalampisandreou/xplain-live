"""Live retake (src/xplain/retake.py, POST /api/live/retake): off by default, guarded, nothing stored, never raises."""
import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from xplain import live, retake
from xplain.api import create_app
from xplain.narrator import MAX_ATTEMPTS
from xplain.retake import Retake, RetakeRefused, find_moment

from .test_narrator import BAD, FakeAgent
from .test_scoring import StubScorer
from .test_verifier import make_bundle, make_out

MOMENT_ID = make_bundle().moment.moment_id


def bundle_dict(n_before: int = 2) -> dict:
    """A served-bundle shape: n_before commentated moments earlier in the match (seq lower), then make_bundle()'s,
    and one not commentated."""
    target = make_bundle().model_dump(mode="json")
    target.update(qualifies_live=True, qualified_by=["ranking"])
    moments = []
    for i in range(n_before):
        b = json.loads(json.dumps(target))
        b["moment"].update(moment_id=f"moment:m1-0000{i}", seq=i)
        moments.append({"bundle": b})
    off = json.loads(json.dumps(target))
    off["moment"].update(moment_id="moment:m1-00001x", seq=1)
    off.update(qualifies_live=False, qualified_by=[])
    return {"meta": {"phrase_mode": "hints"}, "moments": [{"bundle": target}, *moments, {"bundle": off}]}


@pytest.fixture
def served(tmp_path):
    path = tmp_path / "bundle_m1.json"
    path.write_text(json.dumps(bundle_dict()), encoding="utf-8")
    return tmp_path


class FakeCredential:
    def __init__(self, error: Exception | None = None):
        self.error, self.calls = error, 0

    def get_token(self, *scopes):   # sync, like azure.identity's DefaultAzureCredential (run in a thread)
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(token="entra-access-token")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def runner(*answers, credential: FakeCredential | None = None, clock: Clock | None = None, seen: list | None = None,
           **kw) -> Retake:
    agent = FakeAgent(*answers)
    cred = credential or FakeCredential()

    def agent_factory(token, model, mode, backend):
        if seen is not None:
            seen.append((token, model, mode, backend))
        return agent
    return Retake(credential_factory=lambda: cred, agent_factory=agent_factory, clock=clock or Clock(), **kw)


def run(r: Retake, served, moment_id: str = MOMENT_ID) -> dict:
    return asyncio.run(r.run(served / "bundle_m1.json", moment_id))


# --- the moment ---------------------------------------------------------------------------------

def test_the_slot_is_the_index_among_commentated_moments_by_seq():
    moment, slot = find_moment(bundle_dict(n_before=2), MOMENT_ID)
    assert moment.moment.moment_id == MOMENT_ID and slot == 2   # two commentated before it; the other one skipped


def test_an_uncommentated_moment_takes_the_slot_of_the_commentated_moments_before_it():
    moment, slot = find_moment(bundle_dict(n_before=2), "moment:m1-00001x")
    assert moment.moment.moment_id == "moment:m1-00001x" and not moment.qualifies_live
    assert slot == 2   # the commentated moments at seq 0 and 1 sort before it; make_bundle()'s comes after


def test_an_unknown_moment_is_refused():
    with pytest.raises(RetakeRefused) as caught:
        find_moment(bundle_dict(), "moment:nope")
    assert caught.value.status == 404


def test_an_uncommentated_moment_is_retaken(served):
    out = run(runner(make_out()), served, "moment:m1-00001x")
    assert out["ok"] is True and out["moment_id"] == "moment:m1-00001x" and out["line"]["en"]


# --- outcomes -----------------------------------------------------------------------------------

def test_an_accepted_line_comes_back_with_the_trace_and_the_final_check(served, monkeypatch):
    monkeypatch.delenv("XPLAIN_RETAKE_DEPLOYMENT", raising=False)
    seen = []
    out = run(runner(make_out(), seen=seen), served)
    assert out["ok"] is True and out["status"] == "accepted" and out["attempts"] == 1
    assert out["line"] == {"en": make_out().line_en, "es": make_out().line_es, "el": make_out().line_el}
    assert [s["agent"] for s in out["steps"]] == ["narrator", "verifier", "verifier"]
    assert out["steps"][-1]["final"] is True and out["steps"][-1]["checks"]
    assert out["steps"][0]["model"] == "gpt-6.1-sol" == retake.DEFAULT_DEPLOYMENT == out["model"]
    assert out["mode"] == "hints" and out["network_retries"] == 0 and out["latency_s"] >= 0
    assert seen[0][0] == "entra-access-token" and isinstance(seen[0][3], live.AsyncIPv4Backend)


def test_the_model_is_the_retake_deployment_not_the_health_checks(served, monkeypatch):
    monkeypatch.setenv("XPLAIN_LIVE_DEPLOYMENT", "gpt-4.1")
    monkeypatch.setenv("XPLAIN_RETAKE_DEPLOYMENT", "gpt-4.1-mini")
    seen = []
    out = run(runner(make_out(), seen=seen), served)
    assert out["model"] == "gpt-4.1-mini" == out["steps"][0]["model"] == seen[0][1]


def test_a_rejected_answer_is_retried_with_the_reasons(served):
    out = run(runner(BAD, make_out()), served)
    assert out["status"] == "retried" and out["attempts"] == 2 and out["ok"] is True
    assert out["reasons"] and out["reasons"][0].startswith("attempt 1: no_numbers")


def test_rejected_every_time_gives_no_line(served):
    out = run(runner(*[BAD] * MAX_ATTEMPTS), served)
    assert out["ok"] is False and out["status"] == "rejected" and out["line"] is None
    assert out["attempts"] == MAX_ATTEMPTS and out["steps"][-1]["agent"] == "verifier"
    assert "stage" not in out


def test_a_foundry_error_is_a_failure_at_the_request_stage(served):
    out = run(runner(ConnectionError("refused")), served)
    assert out["status"] == "failed" and out["stage"] == "request" and out["line"] is None
    assert "ConnectionError" in out["message"]


def test_a_token_failure_is_a_failure_at_the_token_stage_and_never_calls_the_model(served):
    seen = []
    out = run(runner(make_out(), credential=FakeCredential(PermissionError("no identity")), seen=seen), served)
    assert out["status"] == "failed" and out["stage"] == "token" and out["error"] == "PermissionError"
    assert seen == [] and out["attempts"] == 0


def test_secrets_never_come_back(served):
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4In0.c2lnbmF0dXJl"
    out = run(runner(ConnectionError(f"failed with Bearer {jwt}")), served)
    assert jwt not in json.dumps(out)


def test_each_retake_asks_for_the_next_variation_not_the_line_on_air(served, monkeypatch):
    slots = []
    narrate = retake.Narrator.narrate

    async def spy(self, bundle, slot=None):
        slots.append(slot)
        return await narrate(self, bundle, slot)
    monkeypatch.setattr(retake.Narrator, "narrate", spy)
    clock = Clock()
    r = runner(make_out(), make_out(), make_out(), clock=clock)
    for _ in range(2):
        assert run(r, served)["ok"] is True
        clock.t += retake.COOLDOWN_S
    other = "moment:m1-00000"   # another moment counts its own retakes
    assert run(r, served, other)["ok"] is True
    assert slots == [3, 4, 1]   # MOMENT_ID's slot is 2 (+1, +2); the other's is 0 (+1)


def test_nothing_is_stored_in_the_bundle(served):
    before = (served / "bundle_m1.json").read_text(encoding="utf-8")
    run(runner(make_out()), served)
    assert (served / "bundle_m1.json").read_text(encoding="utf-8") == before


# --- guards -------------------------------------------------------------------------------------

def test_a_second_retake_inside_the_cooldown_is_refused_then_allowed(served):
    clock = Clock()
    r = runner(make_out(), make_out(), clock=clock)
    run(r, served)
    clock.t += retake.COOLDOWN_S - 3
    with pytest.raises(RetakeRefused) as caught:
        run(r, served)
    assert caught.value.reason == "cooldown" and caught.value.retry_after_s == 3
    clock.t += 3
    assert run(r, served)["ok"] is True


def test_only_one_retake_at_a_time(served):
    r = runner("slow", make_out())

    async def both():
        first = asyncio.create_task(r.run(served / "bundle_m1.json", MOMENT_ID))
        await asyncio.sleep(0.05)
        with pytest.raises(RetakeRefused) as caught:
            await r.run(served / "bundle_m1.json", MOMENT_ID)
        first.cancel()
        return caught.value
    assert asyncio.run(both()).reason == "busy"


def test_the_daily_cap_counts_failures_and_resets_the_next_day(served):
    clock, day = Clock(), [dt.date(2026, 10, 22)]
    r = runner(ConnectionError("x"), make_out(), make_out(), clock=clock, today=lambda: day[0], cap=1,
               cooldown_s=0)
    run(r, served)
    with pytest.raises(RetakeRefused) as caught:
        run(r, served)
    assert caught.value.reason == "cap"
    day[0] = dt.date(2026, 10, 23)
    assert run(r, served)["ok"] is True


def test_a_refused_moment_does_not_count_against_the_cap(served):
    r = runner(make_out(), cap=1)
    with pytest.raises(RetakeRefused):
        run(r, served, "moment:nope")
    assert run(r, served)["ok"] is True


# --- the endpoint -------------------------------------------------------------------------------

def client(served, r: Retake) -> TestClient:
    return TestClient(create_app(StubScorer(), bundles=served, live_retake=r))


def test_the_endpoint_is_off_unless_xplain_live_is_1(served, monkeypatch):
    monkeypatch.delenv("XPLAIN_LIVE", raising=False)
    with client(served, runner(make_out())) as c:
        assert c.post("/api/live/retake", json={"match": "m1", "moment_id": MOMENT_ID}).status_code == 404


def test_the_endpoint_answers_and_is_never_cached(served, monkeypatch):
    monkeypatch.setenv("XPLAIN_LIVE", "1")
    with client(served, runner(make_out())) as c:
        r = c.post("/api/live/retake", json={"match": "m1", "moment_id": MOMENT_ID})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert r.headers["cache-control"] == "no-store"


def test_the_endpoint_refusals(served, monkeypatch):
    monkeypatch.setenv("XPLAIN_LIVE", "1")
    with client(served, runner(make_out())) as c:
        assert c.post("/api/live/retake", json={"match": "nope", "moment_id": MOMENT_ID}).status_code == 404
        assert c.post("/api/live/retake", json={"match": "../x", "moment_id": MOMENT_ID}).status_code == 422
        assert c.post("/api/live/retake", json={"match": "m1", "moment_id": "moment:nope"}).status_code == 404
        assert c.post("/api/live/retake", json={"match": "m1", "moment_id": MOMENT_ID}).status_code == 200
        r = c.post("/api/live/retake", json={"match": "m1", "moment_id": MOMENT_ID})
    assert r.status_code == 429 and r.json()["refused"] == "cooldown" and int(r.headers["retry-after"]) >= 1


def test_the_real_agent_talks_to_foundry_over_ipv4_with_the_token_as_bearer(monkeypatch):
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    backend = live.AsyncIPv4Backend()
    agent = retake.make_agent("entra-access-token", "gpt-6.1-sol", "hints", backend)
    openai_client = agent.client.client
    assert str(openai_client.base_url) == "https://example.openai.azure.com/openai/v1/"
    assert openai_client.api_key == "entra-access-token" and openai_client.max_retries == 0
    assert openai_client._client._transport._pool._network_backend is backend
    assert openai_client._client.timeout.read == 60.0   # gpt-6.1-sol's per-call timeout (models.MODEL_PARAMS)


def test_the_final_step_is_the_workflows():
    from xplain import workflow
    from xplain.verifier import verify
    verdict = verify(make_out(), make_bundle(), "hints")
    assert retake.final_step(verdict, 2) == workflow.final_step(verdict, 2)
    assert retake.final_step(verdict, 0) == workflow.final_step(verdict, 0)


def test_the_retake_never_imports_the_build_pipeline():
    """The container imports retake.py; workflow.py would pull in the pipeline, SHAP explainer and recap editor."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    code = "import sys, xplain.api; print(any(m in sys.modules for m in ('xplain.workflow', 'xplain.pipeline')))"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "False"


def test_the_credential_is_the_sync_one_from_live_never_azure_identity_aio():
    """azure.identity.aio's managed identity needs aiohttp, which the image does not install (found on the local
    image build: ImportError at the token stage)."""
    assert Retake()._credential_factory is live.make_credential
