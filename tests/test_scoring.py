import json
import math
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from xplain import analytics
from xplain.api import create_app
from xplain.config import AnalyticsConfig
from xplain.features import FEATURE_ORDER
from xplain.generator import generate_match
from xplain.schema import PassEvent, PassInputs
from xplain.scoring import ModelLoadError, ScoreResult, XPassScorer, find_default_model


class StubScorer:
    """Deterministic stand-in for the model: xP depends on start_x only. Records what it was given."""

    model_version = "stub-1"

    def __init__(self):
        self.seen: list = []

    def score(self, inputs: Sequence[PassInputs]) -> list[ScoreResult]:
        self.seen.extend(inputs)
        return [ScoreResult(expected_probability=0.95 - 0.6 * p.start_x / 120.0) for p in inputs]


@pytest.fixture(scope="module")
def seed42_passes() -> list[PassEvent]:
    events, _ = generate_match(42)
    return [e for e in events if e.type == "pass"]


@pytest.fixture
def stub_client():
    stub = StubScorer()
    with TestClient(create_app(stub)) as client:
        yield client, stub


def pass_body(event: PassEvent, **inputs_override) -> dict:
    body = json.loads(event.model_dump_json())
    body["inputs"].update(inputs_override)
    return body


# --- analytics (pure) --------------------------------------------------------------------

def test_difficulty_is_monotone_and_bounded():
    xs = [i / 1000 for i in range(1001)]
    ds = [analytics.difficulty(x) for x in xs]
    assert all(0 <= d <= 100 for d in ds)
    assert all(a >= b for a, b in zip(ds, ds[1:]))  # higher xP -> easier
    assert analytics.difficulty(1.0) == 0 and analytics.difficulty(0.0) == 100
    assert analytics.difficulty(1.2) == 0 and analytics.difficulty(-0.2) == 100


def test_difficulty_band_boundaries():
    cfg = AnalyticsConfig()
    assert analytics.difficulty_band(0) == "routine"
    assert analytics.difficulty_band(cfg.band_moderate_from - 1) == "routine"
    assert analytics.difficulty_band(cfg.band_moderate_from) == "moderate"
    assert analytics.difficulty_band(cfg.band_hard_from - 1) == "moderate"
    assert analytics.difficulty_band(cfg.band_hard_from) == "hard"
    assert analytics.difficulty_band(cfg.band_exceptional_from - 1) == "hard"
    assert analytics.difficulty_band(cfg.band_exceptional_from) == "exceptional"
    assert analytics.difficulty_band(100) == "exceptional"


def test_delta():
    assert analytics.delta(True, 0.8) == pytest.approx(0.2)
    assert analytics.delta(False, 0.8) == pytest.approx(-0.8)


@pytest.mark.parametrize(
    "completed, xp, expected",
    [
        (True, 0.30, "surprise_success"),   # completed, low xP
        (True, 0.35, "surprise_success"),   # boundary is inclusive
        (True, 0.95, None),                 # completed, high xP: expected
        (False, 0.95, "surprise_failure"),  # failed, high xP
        (False, 0.92, "surprise_failure"),  # boundary is inclusive
        (False, 0.30, None),                # failed, low xP: expected
    ],
)
def test_outlier_rules(completed, xp, expected):
    assert analytics.outlier(completed, xp) == expected


# --- API with a stub scorer ----------------------------------------------------------------

def test_score_endpoint_with_stub(stub_client, seed42_passes):
    client, stub = stub_client
    ev = seed42_passes[10]
    r = client.post("/score", json=pass_body(ev))
    assert r.status_code == 200
    body = r.json()
    xp = 0.95 - 0.6 * ev.inputs.start_x / 120.0
    assert body["xp"] == pytest.approx(xp)
    assert body["event_id"] == ev.event_id and body["seq"] == ev.seq and body["t"] == ev.t
    assert body["completed"] == ev.details.completed
    assert body["difficulty"] == analytics.difficulty(xp)
    assert body["difficulty_band"] == analytics.difficulty_band(body["difficulty"])
    assert body["delta"] == pytest.approx(int(ev.details.completed) - xp)
    assert body["drivers"] == [] and body["model_version"] == "stub-1"
    assert stub.seen == [ev.inputs] and all(type(s) is PassInputs for s in stub.seen)


def test_score_rejects_invalid_body(stub_client, seed42_passes):
    client, stub = stub_client
    r = client.post("/score", json=pass_body(seed42_passes[0], start_x=121.0))
    assert r.status_code == 422
    assert stub.seen == []


def test_score_rejects_non_pass_event(stub_client):
    client, _ = stub_client
    r = client.post("/score", json={"type": "shot", "event_id": "x", "match_id": "m", "seq": 1, "t": 1.0,
                                    "period": 1, "team_id": "a", "x": 100, "y": 40, "shot_speed_ms": 20,
                                    "on_target": True, "goal": False})
    assert r.status_code == 422


def test_details_do_not_change_xp(stub_client, seed42_passes):
    client, _ = stub_client
    body = pass_body(seed42_passes[5])
    other = json.loads(json.dumps(body))
    other["details"].update(end_x=119.0, end_y=1.0, length=90.0, ball_speed_ms=40.0,
                            completed=not body["details"]["completed"])
    a, b = client.post("/score", json=body).json(), client.post("/score", json=other).json()
    assert a["xp"] == b["xp"] and a["difficulty"] == b["difficulty"]
    assert a["completed"] != b["completed"]  # only outcome-derived fields change


def test_health_with_stub(stub_client):
    client, _ = stub_client
    assert client.get("/health").json() == {"status": "ok", "model_version": "stub-1", "n_features": 17}


# --- real artifact ---------------------------------------------------------------------------

def _model_paths():
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc}); see models/README.md")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path} / {feature_list_path}; see models/README.md")
    return model_path, feature_list_path


@pytest.fixture(scope="module")
def real_scorer():
    return XPassScorer(*_model_paths())


def test_real_model_scores_all_seed42_passes(real_scorer, seed42_passes):
    results = real_scorer.score([p.inputs for p in seed42_passes])
    assert len(results) == len(seed42_passes)
    assert all(math.isfinite(r.expected_probability) and 0.0 <= r.expected_probability <= 1.0 for r in results)
    assert real_scorer.model_version.endswith(".joblib")


def test_real_model_ignores_details(real_scorer, seed42_passes):
    with TestClient(create_app(real_scorer)) as client:
        body = pass_body(seed42_passes[3])
        other = json.loads(json.dumps(body))
        other["details"].update(end_x=5.0, end_y=75.0, length=80.0, completed=not body["details"]["completed"])
        assert client.post("/score", json=body).json()["xp"] == client.post("/score", json=other).json()["xp"]


def test_health_with_real_model(real_scorer):
    with TestClient(create_app(real_scorer)) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": real_scorer.model_version, "n_features": 17}


def test_feature_list_mismatch_raises(tmp_path):
    model_path, _ = _model_paths()
    swapped = list(FEATURE_ORDER)
    swapped[2], swapped[3] = swapped[3], swapped[2]
    bad = tmp_path / "feature_list.json"
    bad.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="does not match"):
        XPassScorer(model_path, bad)
    with pytest.raises(ModelLoadError, match="not found"):
        XPassScorer(model_path, tmp_path / "missing.json")


def test_find_default_model_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("XPLAIN_MODEL_PATH", str(tmp_path / "m.joblib"))
    monkeypatch.setenv("XPLAIN_FEATURE_LIST", str(tmp_path / "f.json"))
    assert find_default_model() == (tmp_path / "m.joblib", tmp_path / "f.json")


def test_find_default_model_uses_the_configured_path(tmp_path, monkeypatch):
    monkeypatch.delenv("XPLAIN_MODEL_PATH", raising=False)
    monkeypatch.delenv("XPLAIN_FEATURE_LIST", raising=False)
    (tmp_path / "xpass_v2.joblib").write_bytes(b"")
    model, features = find_default_model(tmp_path / "xpass_v2.joblib")
    assert (model, features) == (tmp_path / "xpass_v2.joblib", tmp_path / "feature_list.json")
    with pytest.raises(ModelLoadError, match="python -m xplain.train"):
        find_default_model(tmp_path / "missing.joblib")


def test_default_model_path_is_xpass_v2():
    from xplain.config import XPASS_MODEL_PATH
    assert XPASS_MODEL_PATH.parts[-3:] == ("models", "xpass_v2", "xpass_v2.joblib")
