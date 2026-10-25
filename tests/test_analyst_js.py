"""Parity of the Analyst mode's browser numbers (src/xplain/web/analyst.js) with their Python references.

The Browser computation exception (README, "Run the web page") lets analyst.js compute counts and sums over the bundle's marks at
the match clock; every such function must equal its Python twin (src/xplain/swing.py). node runs analyst.js on a
bundle built from the seed-42 match and the results are compared exactly, for both teams, at several times, in
every scope. On the demo bundle the swing's last value is the recap card's vs expected, and every SHAP panel adds
up as the Explainer stored it.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xplain.bundle import MatchBundle, build_bundle
from xplain.recap import round2, team_passing
from xplain.swing import TIERS, team_marks, team_swing, team_tiers

from .test_bundle import narrated
from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_workflow import GOOD

ANALYST_JS = Path(__file__).resolve().parents[1] / "src" / "xplain" / "web" / "analyst.js"
BUNDLE_3001 = Path(__file__).resolve().parents[1] / "data" / "bundle_seed3001.json"
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node is not on PATH: the analyst.js parity test needs it")
needs_3001 = pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")

# node: load analyst.js, run every case, print the results as JSON
HARNESS = r"""
const fs = require('fs');
const A = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = d.cases.map((c) => ({
  event_ids: A.teamPasses(d.marks, c.team, c.now, c.period).map((m) => m.event_id),
  swing: A.teamSwing(d.marks, c.team, c.now, c.period),
  tiers: A.teamTiers(d.marks, c.team, c.now, c.period),
  surprises: A.surprises(d.moments, c.now, c.period).map((e) => e.bundle.moment.moment_id),
}));
process.stdout.write(JSON.stringify({ tiers: A.TIERS, out }));
"""


@pytest.fixture(scope="module")
def match(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    return build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)


def run_node(tmp_path, marks, moments, cases):
    data = tmp_path / "cases.json"
    data.write_text(json.dumps({"marks": marks, "moments": moments, "cases": cases}), encoding="utf-8")
    done = subprocess.run([NODE, "-e", HARNESS, str(ANALYST_JS), str(data)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def times(b):
    """Before kick-off, mid first half, just before the second half, mid second half, full time."""
    c = b.clock
    start2 = c.second_half_start
    assert start2 is not None
    return [-1.0, start2 / 2, start2 - 0.001, (start2 + c.last_event_t) / 2, c.last_event_t]


def surprise_ids(b, now, period):
    found = [e.bundle.moment for e in b.moments if e.bundle.moment.t <= now
             and (period is None or e.bundle.moment.period == period)]
    return [m.moment_id for m in sorted(found, key=lambda m: m.seq)]


def swing_js(points):
    return [{"t": p.t, "seq": p.seq, "passes": p.passes, "completed": p.completed, "expectedXp": p.expected_xp,
             "vsExpectedXp": p.vs_expected_xp} for p in points]


def tiers_js(rows):
    return [{"tier": r.tier, "passes": r.passes, "completed": r.completed, "expectedXp": r.expected_xp,
             "vsExpectedXp": r.vs_expected_xp} for r in rows]


@needs_node
def test_analyst_numbers_equal_python(match, tmp_path):
    b = match
    teams = [b.meta.home.team_id, b.meta.away.team_id]
    cases = [{"team": team, "now": now, "period": period}
             for team in teams for now in times(b) for period in (None, 1, 2)]
    dumped = b.model_dump(mode="json")
    got = run_node(tmp_path, dumped["marks"], dumped["moments"], cases)
    assert got["tiers"] == list(TIERS)
    assert len(got["out"]) == len(cases)
    points = surprises_seen = 0
    for case, js in zip(cases, got["out"]):
        team, now, period = case["team"], case["now"], case["period"]
        assert js["event_ids"] == [m.event_id for m in team_marks(b.marks, team, now, period)], case
        # exact: both sides add left to right in seq order, so the floats are equal, not just close
        assert js["swing"] == swing_js(team_swing(b.marks, team, now, period)), case
        assert js["tiers"] == tiers_js(team_tiers(b.marks, team, now, period)), case
        assert js["surprises"] == surprise_ids(b, now, period), case
        points += len(js["swing"])
        surprises_seen += len(js["surprises"])
    assert points > 0 and surprises_seen > 0   # the cases reach both, so the comparison means something


@needs_node
def test_unknown_team_and_kick_off_give_nothing(match, tmp_path):
    b = match
    dumped = b.model_dump(mode="json")
    got = run_node(tmp_path, dumped["marks"], dumped["moments"], [
        {"team": "nobody", "now": b.clock.last_event_t, "period": None},
        {"team": b.meta.home.team_id, "now": -1.0, "period": None},
    ])
    empty = [{"tier": t, "passes": 0, "completed": 0, "expectedXp": 0, "vsExpectedXp": 0} for t in TIERS]
    assert got["out"][0] == {"event_ids": [], "swing": [], "tiers": empty, "surprises": surprise_ids(b, b.clock.last_event_t, None)}
    assert got["out"][1] == {"event_ids": [], "swing": [], "tiers": empty, "surprises": []}


def test_swing_ends_where_the_recap_card_does(match):
    """The running vs expected after a team's last pass of the scope is recap.team_passing's, before round2; the
    tiers split the same passes."""
    b = match
    for team in (b.meta.home, b.meta.away):
        for scope, period in (("full_time", None), ("half_time", 1)):
            card = team_passing(b.marks, team.team_id, team.name, scope)
            swing = team_swing(b.marks, team.team_id, None, period)
            assert (swing[-1].passes, swing[-1].completed) == (card.passes, card.completed)
            assert round2(swing[-1].vs_expected_xp) == card.vs_expected_xp
            assert round2(swing[-1].expected_xp) == card.expected_xp
            tiers = team_tiers(b.marks, team.team_id, None, period)
            assert sum(r.passes for r in tiers) == card.passes and sum(r.completed for r in tiers) == card.completed
    # the running value moves by 1 - xP on a completed pass and by -xP on a failed one
    s = team_swing(b.marks, b.meta.home.team_id)
    for a, z in zip(s, s[1:]):
        m = next(m for m in b.marks if m.seq == z.seq)
        assert z.vs_expected_xp - a.vs_expected_xp == pytest.approx(m.delta)


@needs_3001
def test_demo_bundle_swing_equals_the_card_and_shap_adds_up():
    b = MatchBundle.model_validate_json(BUNDLE_3001.read_text(encoding="utf-8"))
    card = b.recaps["full_time"].card
    for team, side in ((b.meta.home, card.home), (b.meta.away, card.away)):
        assert round2(team_swing(b.marks, team.team_id)[-1].vs_expected_xp) == side.vs_expected_xp
    # the SHAP panel shows the stored values: baseline + drivers + rest = raw score (contracts.Explanation)
    for e in b.moments:
        x = e.bundle.explanation
        assert x.baseline + sum(d.shap for d in x.drivers) + x.rest == pytest.approx(x.raw_score, abs=1e-4)
        assert all((d.shap > 0) == (d.direction == "raises") for d in x.drivers)


def test_analyst_js_names_its_python_references():
    js = ANALYST_JS.read_text(encoding="utf-8")
    assert "Browser computation exception" in js and "swing.team_swing" in js and "swing.team_tiers" in js
    assert "fetch(" not in js and "XMLHttpRequest" not in js   # nothing computed here leaves the page
    assert "XP " not in js and "'XP" not in js
