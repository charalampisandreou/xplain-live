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
    """Like DefaultAzureCredential: a chain that records the credential that returned a token."""
    def __init__(self, error: Exception | None = None):
        self.credentials = [ManagedIdentityCredential()]
        self._successful_credential = None
        self.error, self.scopes = error, []

    def get_token(self, *scopes, **kw):
        self.scopes.append(scopes)
        if self.error:
            raise self.error
        self._successful_credential = self.credentials[0]
        return SimpleNamespace(token="fake-token", expires_on=int(time.time()) + 3600)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def checker(completions: FakeCompletions, clock: Clock | None = None, credential: FakeDefault | None = None,
            tokens: list | None = None, **kw) -> LiveCheck:
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    def client_factory(token):
        if tokens is not None:
            tokens.append(token)
        return client
    cred = credential or FakeDefault()
    kw.setdefault("network", lambda: dict(FAKE_NETWORK))   # no real network in tests
    return LiveCheck(credential_factory=lambda: cred, client_factory=client_factory, clock=clock or Clock(), **kw)


FAKE_NETWORK = {"dns": {}, "resolv": None, "proxy": []}


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
    assert isinstance(body["latency_s"], float) and body["age_s"] == 0.0
    assert body["endpoint_host"] == live.endpoint().split("/")[2]   # the host only
    assert "stage" not in body and "causes" not in body
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
    assert {k: v for k, v in second.items() if k not in ("cached", "age_s")} == \
        {k: v for k, v in first.items() if k not in ("cached", "age_s")}
    assert second["age_s"] == 59.9
    clock.t += 0.1
    assert check.run()["cached"] is False and fake.calls == 2


def test_a_result_older_than_a_minute_is_never_reused(fake):
    """The deployed bug report: a call ~7 minutes later came back cached. 61 s, 7 minutes: always a new call."""
    clock = Clock()
    check = checker(fake, clock)
    assert check.run()["cached"] is False
    clock.t += 61
    third = check.run()
    assert third["cached"] is False and third["age_s"] == 0.0 and fake.calls == 2
    clock.t += 7 * 60
    assert check.run()["cached"] is False and fake.calls == 3
    clock.t += 1
    assert check.run()["cached"] is True and fake.calls == 3


def test_the_endpoint_is_never_cached_by_a_browser_or_proxy(monkeypatch, fake):
    monkeypatch.setenv("XPLAIN_LIVE", "1")
    with app_client(checker(fake)) as c:
        assert c.get("/api/live/health").headers["cache-control"] == "no-store"


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


def test_credential_name_is_the_one_that_returned_a_token():
    chain = FakeDefault()
    assert credential_name(chain) is None   # configured, but no token returned yet
    chain.get_token(live.COGNITIVE_SCOPE)
    assert credential_name(chain) == "ManagedIdentityCredential"
    assert credential_name(ManagedIdentityCredential()) == "ManagedIdentityCredential"
    assert credential_name(None) is None


def test_the_token_comes_first_and_alone_and_is_the_bearer(fake):
    cred, tokens = FakeDefault(), []
    body = checker(fake, credential=cred, tokens=tokens).run()
    assert body["ok"] is True and cred.scopes == [(live.COGNITIVE_SCOPE,)] and tokens == ["fake-token"]


def test_a_token_failure_is_reported_as_the_token_stage_with_no_credential(fake):
    from azure.core.exceptions import ClientAuthenticationError
    cred = FakeDefault(error=ClientAuthenticationError("ManagedIdentityCredential: no identity token=abc123 found"))
    body = checker(fake, credential=cred).run()
    assert body["ok"] is False and body["stage"] == "token" and body["credential"] is None
    assert body["error"] == "ClientAuthenticationError" and "abc123" not in body["message"]
    assert fake.calls == 0   # the request never ran


def test_a_request_failure_names_the_credential_that_gave_the_token():
    body = checker(FakeCompletions(error=ConnectionError("refused"))).run()
    assert body["stage"] == "request" and body["credential"] == "ManagedIdentityCredential"
    assert body["error"] == "ConnectionError" and body["causes"] == []


def test_the_client_sends_the_entra_token_not_a_key(monkeypatch):
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    client = live.make_client("entra-access-token")
    assert str(client.base_url) == "https://example.openai.azure.com/openai/v1/"
    assert live.endpoint_host() == "example.openai.azure.com"
    assert client.api_key == "entra-access-token" and client.max_retries == 0 and client.timeout == live.TIMEOUT_S
    assert live.COGNITIVE_SCOPE == "https://cognitiveservices.azure.com/.default"


def test_the_credential_is_the_default_chain_with_the_managed_identity_client_id(monkeypatch):
    monkeypatch.setenv("AZURE_CLIENT_ID", "00000000-0000-0000-0000-000000000001")
    cred = live.make_credential()
    assert type(cred).__name__ == "DefaultAzureCredential"
    names = [type(c).__name__ for c in cred.credentials]
    assert "ManagedIdentityCredential" in names and "AzureCliCredential" in names


def test_the_cause_chain_is_shown_scrubbed_and_logged(caplog):
    """The openai client wraps any send failure as "Connection error."; the causes are what help."""
    try:
        try:
            try:
                raise OSError("[Errno -2] Name or service not known api-key=abc123")
            except OSError as root:
                raise TimeoutError("connect: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOi.eyJhdWQiOiJodHRw.c2ln") from root
        except TimeoutError as mid:
            raise RuntimeError("Connection error.") from mid
    except RuntimeError as wrapped:
        error = wrapped
    with caplog.at_level("WARNING", logger="xplain.live"):
        body = checker(FakeCompletions(error=error)).run()
    assert body["ok"] is False and body["error"] == "RuntimeError" and body["message"] == "Connection error."
    assert [c["error"] for c in body["causes"]] == ["TimeoutError", "OSError"]
    assert body["causes"][1]["message"].startswith("[Errno -2] Name or service not known")
    logged = caplog.text
    assert "at request: RuntimeError: Connection error. <- TimeoutError: connect:" in logged
    for leaked in ("eyJ", "abc123"):
        assert leaked not in str(body["causes"]) and leaked not in logged
    assert "Traceback" not in logged


def test_the_cause_chain_stops_after_three_and_follows_context():
    def nest(n):
        if n == 0:
            raise ValueError("root")
        try:
            nest(n - 1)
        except Exception:
            raise KeyError(f"level {n}")   # implicit __context__
    try:
        nest(5)
    except KeyError as e:
        error = e
    body = checker(FakeCompletions(error=error)).run()
    assert len(body["causes"]) == live.CAUSE_DEPTH == 3
    assert all(c["error"] == "KeyError" for c in body["causes"])


# --- network diagnosis: dns / resolv / proxy (fakes only, no real lookups) ---

def fake_resolver(answers: dict):
    def resolve(host):
        a = answers[host]
        if isinstance(a, Exception):
            raise a
        if a == "hang":
            time.sleep(1.0)
            return ["192.0.2.99"]
        return a
    return resolve


def test_dns_report_gives_ok_the_first_two_addrs_or_the_error():
    import socket
    resolve = fake_resolver({
        "a.example": ["192.0.2.1", "192.0.2.2", "192.0.2.3"],
        "b.example": socket.gaierror(-3, "Temporary failure in name resolution"),
    })
    out = live.dns_report(["a.example", "b.example", "a.example", None], resolve)
    assert list(out) == ["a.example", "b.example"]   # in order, no repeats, no empty host
    assert out["a.example"] == {"ok": True, "addrs": ["192.0.2.1", "192.0.2.2"], "error": None}
    assert out["b.example"] == {"ok": False, "addrs": [],
                                "error": "gaierror: [Errno -3] Temporary failure in name resolution"}


def test_dns_report_has_one_budget_for_all_hosts():
    resolve = fake_resolver({"slow.example": "hang", "fast.example": ["192.0.2.1"]})
    start = time.perf_counter()
    out = live.dns_report(["slow.example", "fast.example"], resolve, timeout_s=0.2)
    assert time.perf_counter() - start < 0.8
    assert out["slow.example"] == {"ok": False, "addrs": [], "error": "TimeoutError: no answer within 0.2 s"}
    assert out["fast.example"]["ok"] is True


def test_network_report_looks_up_the_endpoint_host_and_the_fixed_hosts(monkeypatch, tmp_path):
    monkeypatch.setenv("XPLAIN_FOUNDRY_ENDPOINT", "https://example.openai.azure.com/openai/v1/")
    asked = []

    def resolve(host):
        asked.append(host)
        return ["192.0.2.1"]
    report = live.network_report(resolve, str(tmp_path / "missing"), environ={})
    assert sorted(asked) == sorted(["example.openai.azure.com", *live.DNS_HOSTS])
    assert list(report["dns"]) == ["example.openai.azure.com", *live.DNS_HOSTS]
    assert live.DNS_HOSTS == ("plhack-foundry-candreou-resource.cognitiveservices.azure.com",
                              "login.microsoftonline.com", "management.azure.com", "pypi.org")
    assert report["resolv"] is None and report["proxy"] == []


def test_resolv_keeps_only_nameserver_search_and_options(tmp_path):
    conf = tmp_path / "resolv.conf"
    conf.write_text("# generated\nnameserver 10.0.0.10\nsearch k8se-apps.svc.cluster.local svc.cluster.local\n"
                    "options ndots:5\ndomain cluster.local\nsortlist 10.0.0.0\n", encoding="utf-8")
    assert live.resolv_lines(str(conf)) == ["nameserver 10.0.0.10",
                                            "search k8se-apps.svc.cluster.local svc.cluster.local", "options ndots:5"]


def test_proxy_names_only_never_values():
    env = {"HTTPS_PROXY": "http://user:secret@proxy:3128", "no_proxy": "localhost", "HTTP_PROXY": ""}
    names = live.proxy_names(env)
    assert names == ["HTTPS_PROXY", "NO_PROXY"]
    assert "secret" not in str(names)


def test_the_diagnosis_is_in_the_body_and_cached_with_the_call(fake):
    runs = []

    def network():
        runs.append(1)
        return {"dns": {"h.example": {"ok": True, "addrs": ["192.0.2.1"], "error": None}},
                "resolv": ["nameserver 10.0.0.10"], "proxy": []}
    clock = Clock()
    check = checker(fake, clock, network=network)
    first = check.run()
    assert first["dns"]["h.example"]["ok"] is True and first["resolv"] == ["nameserver 10.0.0.10"]
    assert first["proxy"] == [] and first["ok"] is True
    clock.t += 30
    assert check.run()["dns"] == first["dns"] and len(runs) == 1   # same 60 s cache as the call
    clock.t += 31
    check.run()
    assert len(runs) == 2


def test_the_diagnosis_comes_with_a_failed_call_too():
    body = checker(FakeCompletions(error=ConnectionError("refused"))).run()
    assert body["ok"] is False and {"dns", "resolv", "proxy"} <= set(body)
