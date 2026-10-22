"""Line check (src/xplain/line_check.py, POST /api/line/check): a producer's line runs the Verifier's fact checks only,
always on, nothing stored. The AI's checks (verifier.CHECKS) are not changed by it."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from xplain import line_check, verifier
from xplain.api import create_app
from xplain.line_check import MAX_CHARS, check_line, inferred_drivers
from xplain.retake import RetakeRefused, find_moment

from .test_scoring import StubScorer

DATA = Path(__file__).resolve().parents[1] / "data"
BUNDLE_3001 = DATA / "bundle_seed3001.json"
pytestmark = pytest.mark.skipif(not BUNDLE_3001.is_file(), reason="data/bundle_seed3001.json is not here")

M41 = "moment:m3001-00758"    # 41', Nevek Iversani (Duskmere United), completed, xP 28%, PVA +72%, difficult
UNCOMMENTATED = "moment:m3001-00680"   # 37', Yanir Iversmont, not commentated, no line


@pytest.fixture(scope="module")
def bundle() -> dict:
    with BUNDLE_3001.open(encoding="utf-8") as f:
        return json.load(f)


def stored(bundle: dict, moment_id: str = M41) -> dict:
    m = next(m for m in bundle["moments"] if m["bundle"]["moment"]["moment_id"] == moment_id)
    return {lang: m["line"][f"line_{lang}"] for lang in verifier.LANGS}


def check(bundle: dict, moment_id: str = M41, **edits: str) -> dict:
    return check_line(bundle, moment_id, {**stored(bundle), **edits})


def failed(result: dict) -> set[str]:
    return {r.split(":", 1)[0] for r in result["reasons"]}


# --- the AI's checks stay as they are -------------------------------------------------------------

def test_the_ai_checks_are_unchanged():
    assert len(verifier.checks_for("hints")) == 18
    for lang in verifier.LANGS:   # BANNED starts with its claim families, then the rest as before
        assert verifier.BANNED[lang][:len(verifier.BANNED_FACTS[lang])] == verifier.BANNED_FACTS[lang]
    assert "numbers" not in verifier.CHECK_LABELS and "names" not in verifier.CHECK_LABELS
    assert not hasattr(verifier, "HUMAN_CHECKS")


def test_a_human_line_runs_facts_only():
    names = [c.name for c in line_check.HUMAN_CHECKS]
    assert names == ["filled", "numbers", "names", "difficulty", "banned", "driver_direction", "outcome_cause",
                     "unsupported_fact", "lead_driver", "length"]
    for phrasing in ("sentence", "no_numbers", "phrases", "role_parity", "one_position_piece", "one_role_per_clause",
                     "tier_first_contrast", "rating_tense", "double_contrast", "needs_driver", "drivers",
                     "driver_expressed"):
        assert phrasing not in names


# --- check_line -----------------------------------------------------------------------------------

def test_the_stored_41_line_passes(bundle):
    r = check(bundle)
    assert r["ok"] and r["reasons"] == []
    assert r["drivers_used"] == [1]   # "lofted high": inferred from the wording
    assert r["line"] == stored(bundle)
    assert r["checks"] == [c.name for c in line_check.HUMAN_CHECKS]
    assert r["labels"]["numbers"] == "Numbers match the data"


def test_every_stored_ai_line_passes_the_fact_checks(bundle):
    for m in bundle["moments"]:
        if m.get("line"):
            r = check_line(bundle, m["bundle"]["moment"]["moment_id"], stored(bundle, m["bundle"]["moment"]["moment_id"]))
            assert r["ok"], (m["bundle"]["moment"]["moment_id"], r["reasons"])


@pytest.mark.parametrize("line", [
    "Nevek Iversani found a teammate with a difficult 28% pass.",          # xP
    "Nevek Iversani's difficult pass came off, 72% above what was expected.",   # PVA
    "In the 41st minute, Nevek Iversani's difficult pass found a teammate.",    # the minute
])
def test_numbers_that_match_the_data_pass(bundle, line):
    r = check(bundle, en=line)
    assert r["ok"], r["reasons"]


@pytest.mark.parametrize("line, said", [
    ("Nevek Iversani found a teammate with a difficult 59% pass.", "'59%'"),
    ("Nevek Iversani's difficult pass found a teammate, 95% of the time.", "'95%'"),
    ("In the 42nd minute, Nevek Iversani's difficult pass found a teammate.", "'42'"),
])
def test_a_wrong_number_fails(bundle, line, said):
    r = check(bundle, en=line)
    assert not r["ok"] and failed(r) == {"numbers"}
    assert said in r["reasons"][0]


def test_another_players_name_fails(bundle):
    r = check(bundle, en="Torian Kestley's difficult pass reaches a teammate.")
    assert failed(r) == {"names"}
    assert "'Torian Kestley'" in r["reasons"][0]


def test_another_club_fails_and_the_passers_own_pass(bundle):
    assert failed(check(bundle, en="Marrowfield's difficult pass reaches a teammate.")) == {"names"}
    r = check(bundle, en="Duskmere United's Iversani plays a difficult pass that reaches a teammate.")
    assert r["ok"], r["reasons"]


def test_easy_on_a_hard_moment_fails(bundle):
    r = check(bundle, en="It was easy, and Nevek Iversani's pass reaches a teammate.")
    assert failed(r) == {"difficulty"}


def test_a_long_line_is_a_note_never_a_reason(bundle):
    r = check(bundle, en="Nevek Iversani's difficult pass " + "really " * 30 + "reaches a teammate.")
    assert r["ok"]
    assert any(n.startswith("length: line_en has") and "may not fit the plate" in n for n in r["notes"])


def test_a_claim_with_no_data_fails(bundle):
    r = check(bundle, en="It was difficult because it was lofted high, but Nevek Iversani's pass reaches a teammate "
                         "in the box.")
    assert failed(r) == {"banned"}


def test_style_words_and_two_sentences_are_the_humans_call(bundle):
    r = check(bundle, en="It was difficult because it was lofted high. Nevek Iversani manages to find a teammate, "
                         "a success.")
    assert r["ok"], r["reasons"]


def test_a_driver_the_moment_does_not_have_fails(bundle):
    r = check(bundle, en="It was difficult because it was played along the ground, but Nevek Iversani's pass reaches "
                         "a teammate.")
    assert failed(r) == {"unsupported_fact"}


def test_a_reason_for_the_outcome_fails(bundle):
    r = check(bundle, en="Nevek Iversani's pass reaches a teammate because it was lofted high, a difficult one.")
    assert "outcome_cause" in failed(r)


def test_mainly_for_a_weaker_driver_fails(bundle):
    r = check_line(bundle, M41, {   # no line names driver 1 ("lofted high"); "mainly" singles out driver 3
        "en": "It was difficult, mainly under pressure, but Nevek Iversani's pass reaches a teammate.",
        "es": "Era difícil, pero el pase de Nevek Iversani llega a un compañero.",
        "el": "Ήταν δύσκολη, αλλά η πάσα του Nevek Iversani φτάνει σε συμπαίκτη."})
    assert r["drivers_used"] == [3]
    assert failed(r) == {"lead_driver"}


def test_an_empty_line_fails(bundle):
    r = check(bundle, es="   ")
    assert failed(r) == {"filled"}
    assert r["reasons"] == ["filled: line_es is empty"]


def test_whitespace_is_collapsed_to_one_line(bundle):
    r = check(bundle, en="It was difficult because it was lofted high,\n but  Nevek Iversani's pass reaches a teammate.")
    assert r["ok"]
    assert r["line"]["en"] == "It was difficult because it was lofted high, but Nevek Iversani's pass reaches a teammate."


def test_an_uncommentated_moment_can_be_written(bundle):
    lines = {   # the stored 80' line's wording, about this moment's passer
        "en": "It was easy because it went along the ground, but Yanir Iversmont's pass goes astray.",
        "es": "Era fácil porque iba raso, pero el pase de Yanir Iversmont no llega a su destino.",
        "el": "Ήταν εύκολη επειδή ήταν συρτή, αλλά η πάσα του Yanir Iversmont δεν καταλήγει πουθενά."}
    r = check_line(bundle, UNCOMMENTATED, lines)
    assert r["ok"], r["reasons"]
    assert r["drivers_used"] == [1]


def test_unknown_moment_is_refused(bundle):
    with pytest.raises(RetakeRefused) as e:
        check_line(bundle, "moment:nope", stored(bundle))
    assert e.value.status == 404


def test_steps_are_a_producer_step_and_the_final_check(bundle):
    steps = check(bundle, en="It was easy, and Nevek Iversani's pass reaches a teammate.")["steps"]
    assert [(s["agent"], s["kind"], s["final"]) for s in steps] == [("producer", "human", False), ("verifier", "code", True)]
    assert steps[1]["ok"] is False and steps[1]["reasons"][0].startswith("difficulty:")
    assert steps[1]["checks"] == [c.name for c in line_check.HUMAN_CHECKS]


def test_drivers_are_inferred_from_wordings(bundle):
    moment, _ = find_moment(bundle, M41)
    assert inferred_drivers(moment, {"en": "lofted high and under pressure", "es": "", "el": ""}) == [1, 3]
    assert inferred_drivers(moment, {"en": "", "es": "", "el": ""}) == []


# --- the endpoint ---------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("XPLAIN_LIVE", raising=False)   # always on: no live switch needed
    with TestClient(create_app(scorer=StubScorer(), bundles=DATA)) as c:
        yield c


def test_endpoint_answers_without_xplain_live(client, bundle):
    res = client.post("/api/line/check", json={"match": "seed3001", "moment_id": M41, "line": stored(bundle)})
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    assert res.json()["ok"] is True


def test_endpoint_reports_a_blocked_line(client, bundle):
    line = {**stored(bundle), "en": "Nevek Iversani's difficult pass, 59%."}
    body = client.post("/api/line/check", json={"match": "seed3001", "moment_id": M41, "line": line}).json()
    assert body["ok"] is False and body["reasons"][0].startswith("numbers:")


def test_endpoint_refuses_unknowns_and_long_lines(client, bundle):
    assert client.post("/api/line/check", json={"match": "seed3001", "moment_id": "moment:nope",
                                                "line": stored(bundle)}).status_code == 404
    assert client.post("/api/line/check", json={"match": "nomatch", "moment_id": M41,
                                                "line": stored(bundle)}).status_code == 404
    long = {**stored(bundle), "en": "a" * (MAX_CHARS + 1)}
    assert client.post("/api/line/check", json={"match": "seed3001", "moment_id": M41, "line": long}).status_code == 422
    assert client.post("/api/line/check", json={"match": "../x", "moment_id": M41,
                                                "line": stored(bundle)}).status_code == 422
