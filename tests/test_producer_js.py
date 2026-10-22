"""Producer mode's log (src/xplain/web/producer.js items()): no spoilers, against a Python twin.

node runs producer.js on bundles and the log at each clock time is compared with the Python twin below: the
moments with t <= now (commentated or not) and the recaps whose time the clock has reached, newest first. Nothing
later ever appears. Producer computes no number: attempts and statuses are the bundle's.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xplain.bundle import build_bundle

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_workflow import GOOD

PRODUCER_JS = Path(__file__).resolve().parents[1] / "src" / "xplain" / "web" / "producer.js"
BUNDLE_3001 = Path(__file__).resolve().parents[1] / "data" / "bundle_seed3001.json"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on PATH: the producer.js test needs it")

HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = d.cases.map((now) => P.items(d.bundle, now, d.recapT).map((i) => [i.type, i.id, i.onAir, i.attempts, i.commentated]));
const fillers = {};
for (const [scope, steps] of Object.entries((d.bundle.trace || {}).recaps || {})) fillers[scope] = P.fillerSummary(steps, 'DUS', 'MAR');
process.stdout.write(JSON.stringify({ items: out, hasTrace: P.hasTrace(d.bundle), fillers, refilled: P.REFILLED,
  empty: P.EMPTY, notRecorded: P.NOT_RECORDED, split: P.splitReason('no_numbers: line_en has 3', { no_numbers: 'No numbers' }) }));
"""


def recap_times(b: dict) -> dict:
    """Where Studio opens each recap (app.js idx.recapT): the last first-half pass, and the last event."""
    out = {}
    h1 = [m["t"] for m in b["marks"] if m["period"] == 1]
    if b.get("recaps", {}).get("half_time") and h1:
        out["half_time"] = h1[-1]
    if b.get("recaps", {}).get("full_time"):
        out["full_time"] = b["clock"]["last_event_t"]
    return out


def twin(b: dict, now: float, recap_t: dict) -> list:
    rows = []
    for m in b["moments"]:
        mo = m["bundle"]["moment"]
        if mo["t"] <= now:
            n = m.get("narration") or {}
            rows.append((mo["t"], 0, ["moment", mo["moment_id"], n.get("status") in ("accepted", "retried"),
                                      n.get("attempts") or 0, m["bundle"]["qualifies_live"]]))
    for scope in ("half_time", "full_time"):
        if scope in recap_t and recap_t[scope] <= now:
            e = b["recaps"][scope]
            rows.append((recap_t[scope], 1, ["recap", scope, e["status"] in ("accepted", "retried"), e["attempts"], True]))
    return [r[2] for r in sorted(rows, key=lambda r: (r[0], r[1]), reverse=True)]


def times(b: dict, recap_t: dict) -> list[float]:
    """Kick-off, just before and at every moment, the recaps' times, full time."""
    ts = [b["clock"]["first_event_t"], b["clock"]["last_event_t"], *recap_t.values()]
    for m in b["moments"]:
        t = m["bundle"]["moment"]["t"]
        ts += [t - 0.001, t]
    return sorted(set(ts))


def run_node(tmp_path, b: dict, cases: list[float]) -> dict:
    data = tmp_path / "cases.json"
    data.write_text(json.dumps({"bundle": b, "recapT": recap_times(b), "cases": cases}), encoding="utf-8")
    done = subprocess.run([NODE, "-e", HARNESS, str(PRODUCER_JS), str(data)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def filler_twin(steps: list, home: str, away: str) -> dict | None:
    """Python twin of fillerSummary: the last recap_filler step's "{slot} = value" notes, as stored."""
    fillers = [s for s in steps if s["agent"] == "recap_filler"]
    if not fillers:
        return None
    slot = dict(n[1:].split("} = ", 1) for n in fillers[-1]["notes"] if n.startswith("{") and "} = " in n)
    parts = [f"{home} vs expected {slot['home_vs_expected']}"] if "home_vs_expected" in slot else []
    parts += [f"{away} vs expected {slot['away_vs_expected']}"] if "away_vs_expected" in slot else []
    if "star_name" in slot:
        parts.append(f"Star {slot['star_name']}" + (f", {slot['star_pva']}" if "star_pva" in slot else ""))
    refill = "filled again by refresh_recap_cards with the card's two-decimal values"
    again = "" if len(fillers) < 2 else (
        "First filled with whole numbers; refilled with the card's two-decimal values." if refill in fillers[-1]["notes"]
        else "Filled again; the last fill is shown.")
    return {"text": " · ".join(parts) or "No numbers in this text.", "again": again}


def check_no_spoilers(tmp_path, b: dict):
    recap_t = recap_times(b)
    cases = times(b, recap_t)
    got = run_node(tmp_path, b, cases)
    for now, items in zip(cases, got["items"]):
        assert items == twin(b, now, recap_t), now
    return got


@pytest.fixture(scope="module")
def built(seed42_events, tiny_scorer):  # noqa: F811
    from .test_workflow import RecapLLM, goalless_recap, run_full
    run_, _ = run_full(seed42_events, tiny_scorer, lambda i, f: GOOD(f), RecapLLM(goalless_recap),
                       scopes=("full_time", "half_time"))
    return build_bundle(seed42_events, tiny_scorer, run_.outcomes, "hints", WIDE, recaps=run_.recaps).model_dump(mode="json")


def test_the_log_holds_only_what_the_clock_reached(tmp_path, built):
    got = check_no_spoilers(tmp_path, built)
    assert got["hasTrace"] is True
    for scope, steps in built["trace"]["recaps"].items():   # one fill, no slots in this fake recap
        assert got["fillers"][scope] == filler_twin(steps, "DUS", "MAR") == {"text": "No numbers in this text.", "again": ""}
    assert got["items"][0] == []                                  # kick-off: nothing yet
    assert [i[0] for i in got["items"][-1]][:1] == ["recap"]      # full time: the full-time recap on top
    assert got["split"] == ["No numbers", "line_en has 3"]


def test_a_retry_is_shown_as_stored_and_never_staged(tmp_path, built):
    b = json.loads(json.dumps(built))
    live = next(m for m in b["moments"] if m["bundle"]["qualifies_live"] and m["line"])
    live["narration"].update(status="retried", attempts=3)
    got = run_node(tmp_path, b, [b["clock"]["last_event_t"]])
    row = next(i for i in got["items"][0] if i[1] == live["bundle"]["moment"]["moment_id"])
    assert row[2:] == [True, 3, True]
    src = PRODUCER_JS.read_text(encoding="utf-8")
    assert "it.attempts > 1 ? 'Retried' : ''" in src


def test_a_bundle_without_a_trace_has_the_empty_state(tmp_path, built):
    b = json.loads(json.dumps(built))
    b.pop("trace")
    got = run_node(tmp_path, b, [0.0])
    assert got["hasTrace"] is False
    assert got["empty"] == "This match was built without a record of the AI agents." and got["notRecorded"].endswith(".")


@pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")
def test_no_spoilers_on_the_demo_bundle(tmp_path):
    b = json.loads(BUNDLE_3001.read_text(encoding="utf-8"))
    got = check_no_spoilers(tmp_path, b)
    assert len(got["items"][-1]) == 22 + 2
    assert sum(not i[4] for i in got["items"][-1]) == 5   # the five surprises that were not commentated
    # the recap's numbers in plain words, from the last fill (the text on air); the 2026-10-20 rebuild filled each
    # text once, with the two-decimal values, so there is no refill line
    assert got["fillers"]["full_time"] == {
        "text": "DUS vs expected 19.45 · MAR vs expected 19.11 · Star Nevek Iversani, 5.48", "again": ""}
    for scope, steps in b["trace"]["recaps"].items():
        assert got["fillers"][scope] == filler_twin(steps, "DUS", "MAR")


RETAKE_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(d.results.map((r) => ({
  steps: P.retakeSteps(d.stored, r).map((s) => [s.agent, Boolean(s.fromMatch), Boolean(s.final)]),
  summary: P.retakeSummary(r), outcome: P.retakeOutcome(r, d.lang), uncommentated: P.retakeOutcome(r, d.lang, false) }))));
"""


def test_a_live_retake_is_shown_as_the_server_sent_it(tmp_path):
    """The stored Analyst and Explainer steps (from the match), then the server's steps; numbers as sent."""
    stored = [{"agent": a, "kind": k, "ok": True} for a, k in
              (("analyst", "code"), ("explainer", "code"), ("narrator", "llm"), ("verifier", "code"))]
    live_steps = [{"agent": "narrator", "kind": "llm", "ok": True}, {"agent": "verifier", "kind": "code", "ok": True},
                  {"agent": "verifier", "kind": "code", "ok": True, "final": True}]
    accepted = {"status": "accepted", "model": "gpt-4.1-mini", "latency_s": 1.42, "attempts": 1, "network_retries": 0,
                "line": {"en": "E", "es": "S", "el": "Λ"}, "steps": live_steps}
    retried = {**accepted, "status": "retried", "attempts": 2, "network_retries": 1}
    rejected = {**accepted, "status": "rejected", "attempts": 3, "line": None, "reasons": ["attempt 1: no_numbers: x"]}
    failed = {"status": "failed", "model": "gpt-4.1-mini", "latency_s": 0.2, "attempts": 0, "network_retries": 0,
              "line": None, "steps": [], "stage": "token", "message": "no identity"}
    path = tmp_path / "retake.json"
    path.write_text(json.dumps({"stored": stored, "results": [accepted, retried, rejected, failed], "lang": "el"},
                               ensure_ascii=False), encoding="utf-8")
    out = subprocess.run([NODE, "-e", RETAKE_HARNESS, str(PRODUCER_JS), str(path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    a, r, j, f = json.loads(out.stdout)
    assert a["steps"] == [["analyst", True, False], ["explainer", True, False], ["narrator", False, False],
                          ["verifier", False, False], ["verifier", False, True]]
    assert a["summary"] == "gpt-4.1-mini · 1.42 s · 1 attempt"
    assert a["outcome"] == {"text": "Λ", "note": "Not on air yet: put it on air, or keep the stored line.", "reasons": []}
    # an uncommentated moment has no stored line to keep
    assert a["uncommentated"] == {"text": "Λ", "note": "Not on air yet: put it on air, or keep it off air.", "reasons": []}
    assert j["uncommentated"] == j["outcome"] and f["uncommentated"] == f["outcome"]
    assert r["summary"] == "gpt-4.1-mini · 1.42 s · 2 attempts · 1 network retry"
    assert j["outcome"]["text"] is None and j["outcome"]["note"] == "Not aired: the Verifier rejected all 3 attempts."
    assert j["outcome"]["reasons"] == ["attempt 1: no_numbers: x"]
    assert f["steps"] == [["analyst", True, False], ["explainer", True, False]]
    assert f["outcome"] == {"text": None, "note": "Retake failed at the token stage.", "reasons": ["no identity"]}
    assert f["summary"] == "gpt-4.1-mini · 0.2 s"


FOLD_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(d.map((steps) => P.foldFinal(steps).map((s) => [s.agent, Boolean(s.final), Boolean(s.again)]))));
"""


def test_the_final_check_folds_into_the_verifier_row_only_when_both_agree(tmp_path):
    def v(ok=True, final=False, checks=("a", "b"), reasons=()):
        return {"agent": "verifier", "kind": "code", "ok": ok, "final": final, "checks": list(checks),
                "reasons": list(reasons)}
    nar = {"agent": "narrator", "kind": "llm", "ok": True, "reasons": []}
    cases = [
        [nar, v(), v(final=True)],                                         # the usual: one row
        [nar, v(ok=False, reasons=["x"]), nar, v(), v(final=True)],         # a retry: the accepted attempt folds
        [nar, v(), v(ok=False, final=True, reasons=["y"])],                 # the final check disagrees: two rows
        [nar, v(), v(final=True, checks=("a",))],                           # different checks ran: two rows
        [nar, v(final=True)],                                               # a reused line: the final check alone
    ]
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    out = subprocess.run([NODE, "-e", FOLD_HARNESS, str(PRODUCER_JS), str(path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    one, retry, differ, checks, reused = json.loads(out.stdout)
    assert one == [["narrator", False, False], ["verifier", False, True]]
    assert retry == [["narrator", False, False], ["verifier", False, False], ["narrator", False, False],
                     ["verifier", False, True]]
    assert differ == [["narrator", False, False], ["verifier", False, False], ["verifier", True, False]]
    assert checks == differ
    assert reused == [["narrator", False, False], ["verifier", True, False]]


FAILED_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(d.map((s) => P.failedChecks(s))));
"""


def test_show_checks_marks_the_checks_a_failed_verifier_step_names(tmp_path):
    """A check is marked failed (✕) when one of the step's reasons starts with its name; the others keep ✓."""
    steps = [
        {"ok": True, "checks": ["sentence", "length", "no_numbers"], "reasons": []},
        {"ok": False, "checks": ["sentence", "length", "no_numbers"],
         "reasons": ["no_numbers: line_en has '3'", "no_numbers: line_es has '3'", "length: line_el has 30 words, max 24"]},
        {"ok": False, "checks": ["sentence"], "reasons": ["final check: something else", "unknown: not a check that ran"]},
        {"ok": False, "reasons": ["no_numbers: x"]},   # checks not recorded: nothing to mark
    ]
    path = tmp_path / "steps.json"
    path.write_text(json.dumps(steps), encoding="utf-8")
    out = subprocess.run([NODE, "-e", FAILED_HARNESS, str(PRODUCER_JS), str(path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert json.loads(out.stdout) == [[], ["no_numbers", "length"], [], []]


EDIT_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify({
  counts: d.texts.map((t) => P.wordCount(t)), limit: P.WORD_LIMIT,
  changed: d.changed.map(([base, lines]) => P.changedLangs(base, lines)),
  steps: d.edits.map(([base, result, edited, written]) => P.editSteps(base, result, edited, written)),
  summaries: [P.editSummary(['en', 'el'], false), P.editSummary(['en', 'es', 'el'], true)],
  outcomes: d.results.map((r) => P.editOutcome(r)),
  words: { aired: P.AIRED_EDIT, airedNc: P.AIRED_EDIT_NC, passed: P.PASSED, blocked: P.BLOCKED } }));
"""


def run_edit(tmp_path, data: dict) -> dict:
    path = tmp_path / "edit.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = subprocess.run([NODE, "-e", EDIT_HARNESS, str(PRODUCER_JS), str(path)], capture_output=True, text=True,
                         encoding="utf-8", check=True)
    return json.loads(out.stdout)


def test_the_edit_helpers(tmp_path):
    """Edit line: the word count is the Verifier's (str.split), only changed languages count as edited, an aired edit's
    steps end with the Producer step and its own final check, and a check's answer reads as stored."""
    from xplain.verifier import WORD_LIMIT
    texts = ["", "   ", "one", "  two  words ", "tab\tand\nnewline here", "Η πάσα του Nevek Iversani φτάνει."]
    stored = [{"agent": "analyst", "kind": "code", "ok": True}, {"agent": "narrator", "kind": "llm", "ok": True},
              {"agent": "verifier", "kind": "code", "ok": True}, {"agent": "verifier", "kind": "code", "ok": True,
                                                                   "final": True}]
    server = [{"agent": "producer", "kind": "human", "ok": True},
              {"agent": "verifier", "kind": "code", "ok": True, "final": True, "checks": ["filled", "numbers"]}]
    result = {"ok": True, "steps": server, "labels": {"filled": "A line in every language"}}
    blocked = {"ok": False, "checks": ["filled", "numbers", "names", "length"],
               "labels": {"numbers": "Numbers match the data", "names": "Only this pass's player and club named"},
               "reasons": ["names: line_en names 'Kestley'", "numbers: line_en has '59%'", "numbers: line_es has '59%'"],
               "notes": ["difficulty: line_en has no difficulty word", "length: line_en has 30 words, max 24; it may not fit the plate"]}
    got = run_edit(tmp_path, {
        "texts": texts,
        "changed": [[{"en": "a b", "es": "x", "el": "y"}, {"en": " a  b ", "es": "x!", "el": "y"}],
                    [None, {"en": "a", "es": "", "el": "c"}]],
        "edits": [[stored, result, ["en"], False], [[], result, ["en", "es", "el"], True]],
        "results": [result, blocked]})
    assert got["counts"] == [len(t.split()) for t in texts]
    assert got["limit"] == WORD_LIMIT
    assert got["changed"] == [["es"], ["en", "el"]]
    edited, written = got["steps"]
    assert [(s["agent"], s.get("final", False)) for s in edited] == [
        ("analyst", False), ("narrator", False), ("verifier", False), ("producer", False), ("verifier", True)]
    assert edited[3]["edited"] == ["en"] and edited[3]["written"] is False
    assert edited[4]["labels"] == {"filled": "A line in every language"}
    assert [s["agent"] for s in written] == ["producer", "verifier"] and written[0]["written"] is True
    assert got["summaries"] == ["Edited by the producer: EN, EL · facts checked by code",
                                "Written by the producer: EN, ES, EL · facts checked by code"]
    ok, no = got["outcomes"]
    assert ok == {"ok": True, "title": "Facts checked: all correct", "fails": [], "warnings": []}
    assert no["title"] == "Blocked: not aired" and no["ok"] is False
    assert no["fails"] == [   # in the order the checks ran, not the order of the reasons
        {"name": "numbers", "label": "Numbers match the data", "details": ["line_en has '59%'", "line_es has '59%'"]},
        {"name": "names", "label": "Only this pass's player and club named", "details": ["line_en names 'Kestley'"]}]
    assert no["warnings"] == ["line_en has 30 words, max 24; it may not fit the plate"]
    assert got["words"]["aired"].startswith("Edited line, put on air") and got["words"]["airedNc"].startswith("Written line")


@pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")
def test_an_edit_check_reads_as_the_server_sent_it(tmp_path):
    """The server's answer (line_check.check_line on the demo bundle) through editOutcome: a blocked line names its
    failing checks by their labels."""
    from xplain.line_check import check_line
    b = json.loads(BUNDLE_3001.read_text(encoding="utf-8"))
    m = next(m for m in b["moments"] if m["bundle"]["moment"]["moment_id"] == "moment:m3001-00758")
    line = {lang: m["line"][f"line_{lang}"] for lang in ("en", "es", "el")}
    good = check_line(b, "moment:m3001-00758", line)
    bad = check_line(b, "moment:m3001-00758", {**line, "en": "Torian Kestley's easy pass, 59%."})
    got = run_edit(tmp_path, {"texts": [], "changed": [], "edits": [], "results": [good, bad]})
    ok, no = got["outcomes"]
    assert ok["ok"] is True and ok["fails"] == []
    assert [f["label"] for f in no["fails"]] == ["Numbers match the data", "Only this pass's player and club named",
                                                 "Difficulty word matches the tier"]
