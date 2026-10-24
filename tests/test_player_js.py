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
from xplain.recap import MINUS, half_up, round2, xp_sum

from .test_bundle import narrated
from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_workflow import GOOD

PLAYER_JS = Path(__file__).resolve().parents[1] / "src" / "xplain" / "web" / "player.js"
BUNDLE_3001 = Path(__file__).resolve().parents[1] / "data" / "bundle_seed3001.json"
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
            # exact: both sides add left to right (analytics.seq_sum), so the floats are equal, not just close
            for k_js, k_py in (("completionRate", "completion_rate"), ("meanXp", "mean_xp"),
                               ("pvaTotal", "pva_total"), ("pvaPerPass", "pva_per_pass")):
                assert s[k_js] == getattr(ref, k_py), (case, k_js)
            # what the recap card shows as Expected and vs expected, before round2: the same floats
            xp = xp_sum(own)   # left to right in seq order, as the page adds
            assert (s["expectedXp"], s["vsExpectedXp"]) == (xp, ref.completed - xp), case
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


FIXED2 = r"""
const P = require(process.argv[1]);
const xs = JSON.parse(process.argv[2]);
process.stdout.write(JSON.stringify(xs.map((x) => [P.fixed2(x, false), P.fixed2(x, true)])));
"""


def fixed2_py(x, signed):
    """The Python twin of player.js fixed2: recap.round2 of |x|, the sign written as the Studio writes it."""
    text = f"{round2(abs(x)):.2f}"
    if text == "0.00":
        return text
    if signed:
        return ("+" if x > 0 else MINUS) + text
    return MINUS + text if x < 0 else text


# exact binary halves (0.125, 0.375), values stored just below a half (1.005, 2.675, 43.515), negatives, zeroes,
# and the demo bundle's card values; each also once already rounded by round2, as the card stores them
VALUES = [0.125, 0.375, -0.125, 1.005, 2.675, 43.515, -19.115, 0.005, -0.005, 0.004, -0.004, 0.0, -0.0,
          393.55, 19.45, -19.11, 5.48, 43.52, 336.11, 1234.565]


def test_fixed2_is_toFixed_and_round2(tmp_path):
    values = VALUES + [round2(x) for x in VALUES]
    done = subprocess.run([NODE, "-e", FIXED2, str(PLAYER_JS), json.dumps(values)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got == [[fixed2_py(x, False), fixed2_py(x, True)] for x in values]
    assert got[0] == ["0.13", "+0.13"] and got[3] == ["1.00", "+1.00"] and got[2] == ["−0.13", "−0.13"]
    assert got[11] == ["0.00", "0.00"] and got[12] == ["0.00", "0.00"]   # no sign on a zero
    # formatting a stored card value again changes nothing
    n = len(VALUES)
    assert all(fixed2_py(round2(x), s) == fixed2_py(x, s) for x in VALUES for s in (False, True))
    assert got[n:] == got[:n]


@pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")
@pytest.mark.parametrize("scope", ["half_time", "full_time"])
def test_star_numbers_in_players_equal_the_recap_card(tmp_path, scope):
    """The star's Player numbers where Studio opens the recap equal the card's star (data/bundle_seed3001.json)."""
    b = json.loads(BUNDLE_3001.read_text(encoding="utf-8"))
    star = b["recaps"][scope]["card"]["star"]
    assert star is not None
    if scope == "full_time":
        now, period = b["clock"]["last_event_t"], None
    else:   # the last first-half pass, scope 1st (app.js idx.recapT.half_time)
        now, period = [m for m in b["marks"] if m["period"] == 1][-1]["t"], 1
    got = run_node(tmp_path, b["marks"], b["moments"], [{"player": star["player_id"], "now": now, "period": period}])
    s = got[0]["stats"]
    assert (s["passes"], s["completed"]) == (star["passes"], star["completed"])
    # two decimals on both screens; the text's whole numbers are those values rounded
    assert (round2(s["expectedXp"]), round2(s["vsExpectedXp"])) == (star["expected_xp"], star["vs_expected_xp"])
    assert half_up(s["expectedXp"]) == star["expected"] and star["completed"] - star["expected"] == star["pva"]


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


# node: fullTimeStar / starShown for each case (bundle, now, player)
STAR_HARNESS = r"""
const fs = require('fs');
const P = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
process.stdout.write(JSON.stringify(d.cases.map((c) => [P.fullTimeStar(d.bundles[c.b], c.now),
                                                        P.starShown(d.bundles[c.b], c.now, c.player)])));
"""


def run_star(tmp_path, bundles, cases):
    data = tmp_path / "star.json"
    data.write_text(json.dumps({"bundles": bundles, "cases": cases}), encoding="utf-8")
    done = subprocess.run([NODE, "-e", STAR_HARNESS, str(PLAYER_JS), str(data)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_star_shows_only_for_the_full_time_star_from_full_time(tmp_path):
    """Player's star: the full-time recap card's star (recap.star_player), from clock.last_event_t on, nobody else."""
    clock = {"last_event_t": 100.0}
    star = {"clock": clock, "recaps": {"half_time": {"card": {"star": {"player_id": "b"}}},
                                       "full_time": {"card": {"star": {"player_id": "a"}}}}}
    no_star = {"clock": clock, "recaps": {"full_time": {"card": {"star": None}}}}
    no_recap = {"clock": clock, "recaps": {}}
    cases = [{"b": 0, "now": 50.0, "player": "a"},     # before full time: no star
             {"b": 0, "now": 99.999, "player": "a"},   # one tick before
             {"b": 0, "now": 100.0, "player": "a"},    # at full time: the star
             {"b": 0, "now": 120.0, "player": "a"},    # after
             {"b": 0, "now": 100.0, "player": "b"},    # the half-time star is not the full-time star
             {"b": 0, "now": 100.0, "player": "c"},    # any other player
             {"b": 0, "now": None, "player": "a"},
             {"b": 1, "now": 100.0, "player": "a"},    # a card with no star
             {"b": 2, "now": 100.0, "player": "a"}]    # a bundle with no full-time recap
    assert run_star(tmp_path, [star, no_star, no_recap], cases) == [
        [None, False], [None, False], ["a", True], ["a", True], ["a", False], ["a", False],
        [None, False], [None, False], [None, False]]


@pytest.mark.skipif(not BUNDLE_3001.exists(), reason="data/ is git-ignored: the demo bundle is local only")
def test_the_demo_matchs_star_is_the_full_time_cards_star(tmp_path):
    b = json.loads(BUNDLE_3001.read_text(encoding="utf-8"))
    ft, star = b["clock"]["last_event_t"], b["recaps"]["full_time"]["card"]["star"]["player_id"]
    cases = [{"b": 0, "now": ft - 0.001, "player": star}, {"b": 0, "now": ft, "player": star}]
    cases += [{"b": 0, "now": ft, "player": p["player_id"]} for p in b["meta"]["players"] if p["player_id"] != star]
    got = run_star(tmp_path, [{"clock": b["clock"], "recaps": b["recaps"]}], cases)
    assert got[0] == [None, False] and got[1] == [star, True]
    assert all(g == [star, False] for g in got[2:]) and len(got) > 20
