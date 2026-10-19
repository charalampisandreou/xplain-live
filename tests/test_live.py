"""Live spike (src/xplain/live.py, GET /api/live/health): off by default, one call per 60 s, secrets never shown."""
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from xplain import live
from xplain.api import create_app
from xplain.live import LiveCheck, credential_name, scrub

from .test_scoring import StubScorer


class FakeCompletions:
    def __init__(self, error: Exception | None = None, delay_s: float = 0.0):
        self.calls, self.error, self.delay_s, self.kwargs = 0, error, delay_s, None

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error:
            raise self.error
        return SimpleNamespace(model="gpt-4.1-mini-2025-04-14")


class ManagedIdentityCredential:   # named like the class DefaultAzureCredential picks in Azure
    pass


class FakeDefault:
    def __init__(self):
        self._successful_credential = ManagedIdentityCredential()


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def checker(completions: FakeCompletions, clock: Clock | None = None, **kw) -> LiveCheck:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LiveCheck(credential_factory=FakeDefault, client_factory=lambda cred: client, clock=clock or Clock(), **kw)


@pytest.fixture
def fake():
    return FakeCompletions()


def app_client(check: LiveCheck) -> TestClient:
    return TestClient(create_app(StubScorer(), live_check=check))


def test_the_endpoint_is_off_unless_xplain_live_is_1(monkeypatch, fake):
    monkeypatch.delenv("XPLAIN_LIVE", raising=False)
    with app_client(checker(fake)) as c:
        assert c.get("/api/live/health").status_code == 404
        monkeypatch.setenv("XPLAIN_LIVE", "true")
        assert c.get("/api/live/health").status_code == 404
    assert fake.calls == 0


def test_one_tiny_call_with_the_live_deployment(monkeypatch, fake):
    monkeypatch.setenv("XPLAIN_LIVE", "1")
    monkeypatch.delenv("XPLAIN_LIVE_DEPLOYMENT", raising=False)
    with app_client(checker(fake)) as c:
        body = c.get("/api/live/health").json()
    assert body["ok"] is True and body["cached"] is False
    assert body["model"] == "gpt-4.1-mini-2025-04-14" and body["credential"] == "ManagedIdentityCredential"
    assert isinstance(body["latency_s"], float)
    assert fake.kwargs == {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Reply with OK"}],
                           "max_completion_tokens": 5}
    monkeypatch.setenv("XPLAIN_LIVE_DEPLOYMENT", "gpt-6.1-sol")
    assert live.deployment() == "gpt-6.1-sol"


def test_at_most_one_real_call_per_minute_cached_in_between(fake):
    clock = Clock()
    check = checker(fake, clock)
    first = check.run()
    clock.t += 59.9
    second = check.run()
    assert fake.calls == 1 and first["cached"] is False and second["cached"] is True
    assert {k: v for k, v in second.items() if k != "cached"} == {k: v for k, v in first.items() if k != "cached"}
    clock.t += 0.1
    assert check.run()["cached"] is False and fake.calls == 2


def test_a_failure_is_rate_limited_too_and_hides_secrets():
    secret = ("Authentication failed: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOi.eyJhdWQiOiJodHRwczovL2NvZ25pdGl2.ZXNlcn "
              "api-key=abcd1234 token: xyz987 sig=QWxhZGRpbjpvcGVuIHNlc2FtZQQWxhZGRpbjpvcGVu")
    fake = FakeCompletions(error=PermissionError(secret))
    clock = Clock()
    check = checker(fake, clock)
    body = check.run()
    assert body["ok"] is False and body["error"] == "PermissionError"
    for leaked in ("eyJ", "abcd1234", "xyz987", "QWxhZGRp", "Bearer e"):
        assert leaked not in body["message"], leaked
    assert body["message"].startswith("Authentication failed:") and len(body["message"]) <= live.MESSAGE_CHARS
    clock.t += 30
    assert check.run()["cached"] is True and fake.calls == 1


def test_a_call_has_a_hard_timeout():
    fake = FakeCompletions(delay_s=0.5)
    body = checker(fake, timeout_s=0.1).run()
    assert body["ok"] is False and body["error"] == "TimeoutError" and body["latency_s"] < 0.5


def test_scrub_keeps_plain_words():
    assert scrub("DefaultAzureCredential failed to retrieve a token from the included credentials.") == \
        "DefaultAzureCredential failed to retrieve a token from the included credentials."
    assert scrub("x\n  y") == "x y"


def test_credential_name():
    assert credential_name(FakeDefault()) == "ManagedIdentityCredential"
    assert credential_name(ManagedIdentityCredential()) == "ManagedIdentityCredential"
    assert credential_name(None) is None


def test_the_client_uses_a_token_provider_not_a_key(monkeypatch):
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    asked = []

    class Cred:
        def get_token(self, *scopes, **kw):
            asked.append(scopes)
            return SimpleNamespace(token="t", expires_on=int(time.time()) + 3600)

    client = live.make_client(Cred())
    assert str(client.base_url) == "https://example.openai.azure.com/openai/v1/"
    assert client.api_key == "" and client.max_retries == 0 and client.timeout == live.TIMEOUT_S
    assert live.COGNITIVE_SCOPE == "https://cognitiveservices.azure.com/.default"


def test_the_credential_is_the_default_chain_with_the_managed_identity_client_id(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000001")
    cred = live.make_credential()
    assert type(cred).__name__ == "DefaultAzureCredential"
    names = [type(c).__name__ for c in cred.credentials]
    assert "ManagedIdentityCredential" in names and "AzureCliCredential" in names
