"""Producer mode's log (src/xplain/web/producer.js items()): no spoilers, against a Python twin.

node runs producer.js on bundles and the log at each clock time is compared with the Python twin below: the
commentated moments with t <= now and the recaps whose time the clock has reached, newest first. Nothing later
ever appears. Producer computes no number: attempts and statuses are the bundle's.
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
const out = d.cases.map((now) => P.items(d.bundle, now, d.recapT).map((i) => [i.type, i.id, i.onAir, i.attempts]));
process.stdout.write(JSON.stringify({ items: out, hasTrace: P.hasTrace(d.bundle),
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
        if m["bundle"]["qualifies_live"] and mo["t"] <= now:
            n = m.get("narration") or {}
            rows.append((mo["t"], 0, ["moment", mo["moment_id"], n.get("status") in ("accepted", "retried"),
                                      n.get("attempts") or 0]))
    for scope in ("half_time", "full_time"):
        if scope in recap_t and recap_t[scope] <= now:
            e = b["recaps"][scope]
            rows.append((recap_t[scope], 1, ["recap", scope, e["status"] in ("accepted", "retried"), e["attempts"]]))
    return [r[2] for r in sorted(rows, key=lambda r: (r[0], r[1]), reverse=True)]


def times(b: dict, recap_t: dict) -> list[float]:
    """Kick-off, just before and at every commentated moment, the recaps' times, full time."""
    ts = [b["clock"]["first_event_t"], b["clock"]["last_event_t"], *recap_t.values()]
    for m in b["moments"]:
        if m["bundle"]["qualifies_live"]:
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
    assert got["items"][0] == []                                  # kick-off: nothing yet
    assert [i[0] for i in got["items"][-1]][:1] == ["recap"]      # full time: the full-time recap on top
    assert got["split"] == ["No numbers", "line_en has 3"]


def test_a_retry_is_shown_as_stored_and_never_staged(tmp_path, built):
    b = json.loads(json.dumps(built))
    live = next(m for m in b["moments"] if m["bundle"]["qualifies_live"] and m["line"])
    live["narration"].update(status="retried", attempts=3)
    got = run_node(tmp_path, b, [b["clock"]["last_event_t"]])
    row = next(i for i in got["items"][0] if i[1] == live["bundle"]["moment"]["moment_id"])
    assert row[2:] == [True, 3]
    src = PRODUCER_JS.read_text(encoding="utf-8")
    assert "it.attempts > 1 ? 'Retried' : ''" in src


def test_a_bundle_without_a_trace_has_the_empty_state(tmp_path, built):
    b = json.loads(json.dumps(built))
    b.pop("trace")
    got = run_node(tmp_path, b, [0.0])
    assert got["hasTrace"] is False
    assert got["empty"] == "No agent trace in this match bundle." and got["notRecorded"].endswith(".")


@pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")
def test_no_spoilers_on_the_demo_bundle(tmp_path):
    b = json.loads(BUNDLE_3001.read_text(encoding="utf-8"))
    got = check_no_spoilers(tmp_path, b)
    assert len(got["items"][-1]) == 17 + 2
