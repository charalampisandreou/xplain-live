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


def test_narration_nav_is_on_the_page(client):
    html = client.get("/").text
    assert 'id="narrnav"' in html and 'hidden' in html.split('id="narrnav"')[1].split(">")[0]
    assert 'id="narr-prev" aria-label="Previous narrated moment"' in html
    assert 'id="narr-next" aria-label="Next narrated moment"' in html
    assert 'id="narr-count" aria-live="polite"' in html


def test_narration_keys_and_replay_are_wired(client):
    js = client.get("/static/app.js").text
    assert "e.key === '['" in js and "narrStep(-1)" in js
    assert "e.key === ']'" in js and "narrStep(1)" in js
    assert "enterReplay(target.bundle.moment.moment_id)" in js   # the chevrons reuse replay mode


def test_goal_state_text_uses_the_minute_label(client):
    js = client.get("/static/app.js").text
    body = js.split("function goalTabText(s)")[1].split("}")[0]
    assert "s.minute_label" in body and "'Goal · '" in body
    # fires only from the playback advance in frame(); the goal plate only while live
    frame = js.split("function frame(ts)")[1].split("function isLive")[0]
    assert "goalFlash(after, ts)" in frame and "state.goal.restartT" in frame
    flash = js.split("function goalFlash(i, ts)")[1].split("function pulse(")[0]
    assert flash.index("pulse(3") < flash.index("if (!isLive()) return") < flash.index("state.goal = {")


def test_xp_keeps_its_casing(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    for text in (html, js):
        assert "XP " not in text and "XP<" not in text and "'XP" not in text
    assert ".xp { text-transform: none; }" in css
    assert '<span class="xp">xP</span>' in html.split('id="lower-xp"')[1]   # the upper-case plate meta
    plate = js.split("function renderPlate(entry)")[1].split("async function fitLines")[0]
    assert "' · xP" not in plate and "'xP" not in plate   # the meta no longer writes xP as text


def test_plate_has_one_fixed_height(client):
    css = client.get("/static/app.css").text
    js = client.get("/static/app.js").text
    body = css.split(".body {")[1].split("}")[0]
    assert "height: 96px" in body and "min-height" not in body
    assert "line-height: 24px" in css.split(".line {")[1].split("}")[0]
    assert "MIN_LINE_PX = 15" in js and "TWO_LINES_PX = 48" in js


def test_plate_transitions_are_light(client):
    js = client.get("/static/app.js").text
    css = client.get("/static/app.css").text
    assert "offsetWidth" not in js   # no forced layout to restart a fade
    plate = js.split("function renderPlate(entry)")[1].split("// --- stage scaling")[0]
    assert "innerHTML" not in plate and "textContent" in plate
    assert "prefers-reduced-motion: reduce" in js and "reducedMotion.matches" in js
    assert "tabular-nums" in css and "min-width: 7ch" in css


def test_pitch_is_pinned_and_plate_columns_are_scoped(client):
    css = client.get("/static/app.css").text
    # a bare .left rule also styled .possession.left and pushed the pitch drawing down
    assert "\n.left {" not in css and "\n.right {" not in css
    assert ".body > .left {" in css and ".body > .right {" in css
    assert "position: absolute; top: 0; left: 0;" in css.split(".pitch > #pitch-svg {")[1].split("}")[0]
    # a rule for every svg in .pitch also caught the possession tab's chevrons
    assert ".pitch svg" not in css


def test_health_and_score_routes_are_kept(client):
    assert client.get("/health").json()["status"] == "ok"
    assert any(getattr(r, "path", None) == "/score" for r in client.app.routes)
