"""The web routes: the Studio page, its static files and the match bundle endpoint."""
import pytest
from fastapi.testclient import TestClient

from xplain import api
from xplain.api import create_app

from .test_scoring import StubScorer


@pytest.fixture
def bundle_dir(tmp_path):
    (tmp_path / "bundle_demo_1.json").write_text('{"meta": {"schema_version": "2"}}', encoding="utf-8")
    (tmp_path / "secret.json").write_text('{"secret": true}', encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(bundle_dir):
    with TestClient(create_app(StubScorer(), bundles=bundle_dir)) as c:
        yield c


def test_index_is_the_studio_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "/static/app.js" in r.text and "/static/app.css" in r.text


def test_static_files_are_served(client):
    js = client.get("/static/app.js")
    css = client.get("/static/app.css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "--bg: #0A1B2E" in css.text


def test_a_bundle_is_served_from_the_bundle_dir(client):
    r = client.get("/api/bundle/demo_1")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json() == {"meta": {"schema_version": "2"}}


def test_missing_bundle_is_404(client):
    assert client.get("/api/bundle/nope").status_code == 404


@pytest.mark.parametrize("name", ["Demo_1", "demo-1", "demo.1", "..%2Fsecret", "%2e%2e", "a b", "x" * 65])
def test_bad_names_never_reach_the_disk(client, name):
    r = client.get("/api/bundle/" + name)
    assert r.status_code in (404, 422)
    assert "secret" not in r.text


def test_bundle_dir_env_override(bundle_dir, monkeypatch):
    monkeypatch.setenv("XPLAIN_BUNDLE_DIR", str(bundle_dir))
    assert api.bundle_dir() == bundle_dir
    with TestClient(create_app(StubScorer())) as c:
        assert c.get("/api/bundle/demo_1").status_code == 200


def test_health_and_score_routes_are_kept(client):
    assert client.get("/health").json()["status"] == "ok"
    assert any(getattr(r, "path", None) == "/score" for r in client.app.routes)
