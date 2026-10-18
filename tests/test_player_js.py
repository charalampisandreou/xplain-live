"""Parity of the Player mode's browser numbers (src/xplain/web/player.js) with their Python references.

The Browser computation exception (CLAUDE.md) lets player.js compute sums, counts and means over the
bundle's marks at the match clock; every such function must equal its Python twin. node runs player.js
on a bundle built from the seed-42 match and the results are compared with pipeline.player_summary
(as_of_t) and the explicit Python equivalents below, for every player, at several times, in every scope.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from xplain.bundle import build_bundle
from xplain.pipeline import player_summary

from .test_bundle import narrated
from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_workflow import GOOD

PLAYER_JS = Path(__file__).resolve().parents[1] / "src" / "xplain" / "web" / "player.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on PATH: the player.js parity test needs it")

# node: load player.js, run every case, print the results as JSON
HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = d.cases.map((c) => {
  const passes = P.playerPasses(d.marks, c.player, c.now, c.period);
  const hardest = P.hardestCompleted(passes);
  return {
    event_ids: passes.map((m) => m.event_id),
    stats: P.playerStats(passes),
    hardest: hardest ? hardest.event_id : null,
    narrated: P.narratedFor(d.moments, c.player, c.now, c.period).map((e) => e.bundle.moment.moment_id),
    tiers: P.tierCounts(passes),
    completedTiers: P.completedTierCounts(passes),
    moments: P.momentsFor(d.moments, c.player, c.now, c.period).map((e) => e.bundle.moment.moment_id),
  };
});
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def match(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    return build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)


def run_node(tmp_path, marks, moments, cases):
    data = tmp_path / "cases.json"
    data.write_text(json.dumps({"marks": marks, "moments": moments, "cases": cases}), encoding="utf-8")
    done = subprocess.run([NODE, "-e", HARNESS, str(PLAYER_JS), str(data)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def times(b):
    """Mid first half, just before the second half, mid second half, full time."""
    c = b.clock
    start2 = c.second_half_start
    assert start2 is not None
    return [start2 / 2, start2 - 0.001, (start2 + c.last_event_t) / 2, c.last_event_t]


def hardest_completed(marks):
    done = [m for m in marks if m.completed]
    return min(done, key=lambda m: (m.xp, m.seq)).event_id if done else None


def narrated_ids(b, player_id, now, period):
    found = [e.bundle.moment for e in b.moments if e.line is not None and e.bundle.moment.player_id == player_id
             and e.bundle.moment.t <= now and (period is None or e.bundle.moment.period == period)]
    return [m.moment_id for m in sorted(found, key=lambda m: m.seq)]


TIERS = ("very easy", "easy", "tricky", "difficult", "very difficult")


def tier_counts(marks):
    return {t: sum(m.tier.en == t for m in marks) for t in TIERS}


def commentated_ids(b, player_id, now, period):
    found = [e.bundle.moment for e in b.moments if e.bundle.qualifies_live and e.bundle.moment.player_id == player_id
             and e.bundle.moment.t <= now and (period is None or e.bundle.moment.period == period)]
    return [m.moment_id for m in sorted(found, key=lambda m: -m.seq)]


def test_player_numbers_equal_python(match, tmp_path):
    b = match
    cases = [{"player": p.player_id, "now": now, "period": period}
             for p in b.meta.players for now in times(b) for period in (None, 1, 2)]
    dumped = b.model_dump(mode="json")
    got = run_node(tmp_path, dumped["marks"], dumped["moments"], cases)
    assert len(got) == len(cases)
    narrated_seen = hardest_seen = 0
    for case, js in zip(cases, got):
        pid, now, period = case["player"], case["now"], case["period"]
        marks = [m for m in b.marks if period is None or m.period == period]
        own = [m for m in sorted(marks, key=lambda m: m.seq) if m.player_id == pid and m.t <= now]
        assert js["event_ids"] == [m.event_id for m in own], case
        ref = player_summary(marks, pid, as_of_t=now)
        if ref is None:
            assert js["stats"] is None, case
        else:
            s = js["stats"]
            assert (s["passes"], s["completed"]) == (ref.passes, ref.completed), case
            for k_js, k_py in (("completionRate", "completion_rate"), ("meanXp", "mean_xp"),
                               ("pvaTotal", "pva_total"), ("pvaPerPass", "pva_per_pass")):
                assert s[k_js] == pytest.approx(getattr(ref, k_py), abs=1e-12), (case, k_js)
        assert js["hardest"] == hardest_completed(own), case
        assert js["narrated"] == narrated_ids(b, pid, now, period), case
        assert js["tiers"] == tier_counts(own), case
        assert js["completedTiers"] == tier_counts([m for m in own if m.completed]), case
        assert js["moments"] == commentated_ids(b, pid, now, period), case
        narrated_seen += bool(js["narrated"])
        hardest_seen += js["hardest"] is not None
    assert narrated_seen > 0 and hardest_seen > 0   # the cases reach both, so the comparison means something


def test_unknown_player_and_kick_off_give_nothing(match, tmp_path):
    b = match
    dumped = b.model_dump(mode="json")
    first = b.meta.players[0].player_id
    got = run_node(tmp_path, dumped["marks"], dumped["moments"], [
        {"player": "nobody", "now": b.clock.last_event_t, "period": None},
        {"player": first, "now": -1.0, "period": None},
    ])
    for js in got:
        assert js == {"event_ids": [], "stats": None, "hardest": None, "narrated": [], "moments": [],
                      "tiers": {"very easy": 0, "easy": 0, "tricky": 0, "difficult": 0, "very difficult": 0},
                      "completedTiers": {"very easy": 0, "easy": 0, "tricky": 0, "difficult": 0, "very difficult": 0}}


def mean(values):
    values = list(values)
    assert values
    return sum(values) / len(values)


def test_pass_geometry_is_the_passers_frame_and_display_flips(match, seed42_events):  # noqa: F811
    """Player mode draws mark.pass_geometry (set in pipeline.py from the event's inputs/details, the
    schema's frame: "the attacking goal centre is at (120, 40)"). In that frame every team attacks
    +x in both halves; display{} is the broadcast pitch, which turns round at half-time."""
    b = match
    meta = next(e for e in seed42_events if type(e).__name__ == "MatchMetaEvent")
    keepers = {p.player_id for p in meta.players if p.position == "GK"}
    shots = [e for e in seed42_events if type(e).__name__ == "ShotEvent"]
    home, away = b.meta.home.team_id, b.meta.away.team_id
    for team in (home, away):
        for period in (1, 2):
            own = [m for m in b.marks if m.team_id == team and m.period == period]
            g = [m.pass_geometry for m in own]
            assert mean(p.end_x - p.start_x for p in g) > 0, (team, period)   # forward on average: toward +x
            gk = [m.pass_geometry.start_x for m in own if m.player_id in keepers]
            assert mean(gk) < 40, (team, period)                              # the keeper plays from x near 0
            team_shots = [s.x for s in shots if s.team_id == team and s.period == period]
            if team_shots:
                assert mean(team_shots) > 80, (team, period)                  # shots go toward x = 120
            # display: unchanged where the team attacks right on the broadcast pitch, turned round elsewhere
            right = b.meta.attacks_right[str(period)] == team
            dx = mean(m.display.end_x - m.display.start_x for m in own)
            assert (dx > 0) == right, (team, period)
            for m in own:
                same = (m.display.start_x, m.display.end_x) == (m.pass_geometry.start_x, m.pass_geometry.end_x)
                assert same == right, m.event_id


def test_player_js_names_its_python_references():
    js = PLAYER_JS.read_text(encoding="utf-8")
    assert "Browser computation exception" in js and "player_summary" in js
    assert "fetch(" not in js and "XMLHttpRequest" not in js   # nothing computed here leaves the page
