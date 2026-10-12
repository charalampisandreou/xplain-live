import json

import pytest
from pydantic import ValidationError

from xplain import analytics
from xplain.analyst import build_moment, qualifies_live, select_half_highlights, select_highlights
from xplain.config import AnalyticsConfig
from xplain.contracts import Moment
from xplain.schema import BodyPart, PassDetails, PassEvent, PassHeight, PassInputs, PlayPattern

CFG = analytics.DEFAULT
SUCCESS_XP = CFG.surprise_success_max_xp   # completed at this xP -> surprise_success
FAILURE_XP = CFG.surprise_failure_min_xp   # failed at this xP -> surprise_failure
ROUTINE_XP = (SUCCESS_XP + FAILURE_XP) / 2  # between the thresholds -> never an outlier

EVIDENCE_NAMES = ("xp", "delta", "difficulty", "difficulty_band", "completed", "minute", "minute_added",
                  "minute_label", "length", "ball_speed_ms", "start_x", "start_y", "end_x", "end_y")


def make_event(seq: int = 10, *, completed: bool = True, t: float = 600.0, period: int = 1,
               team_id: str = "brk", player_id: str | None = "brk_p7") -> PassEvent:
    return PassEvent(
        event_id=f"m1:{seq}", match_id="m1", seq=seq, t=t, period=period,
        team_id=team_id, player_id=player_id,
        inputs=PassInputs(start_x=40.5, start_y=22.25, height=PassHeight.GROUND, body_part=BodyPart.FOOT,
                          play_pattern=PlayPattern.REGULAR, under_pressure=False),
        details=PassDetails(end_x=61.0, end_y=30.75, length=22.19, ball_speed_ms=15.3, completed=completed,
                            recipient_id="brk_p9"),
    )


SECOND_HALF_START = 2850.0  # a first half with 2.5 minutes of added time


def moment(seq: int, xp: float, completed: bool, **kw) -> Moment:
    m = build_moment(make_event(seq, completed=completed, **kw), xp, "test-model", second_half_start=SECOND_HALF_START)
    assert m is not None
    return m


# --- build_moment -------------------------------------------------------------------------------

@pytest.mark.parametrize("completed", [True, False])
def test_non_outlier_returns_none(completed):
    assert build_moment(make_event(completed=completed), ROUTINE_XP, "test-model") is None


def test_surprise_success_has_positive_delta():
    m = moment(1, SUCCESS_XP, completed=True)
    assert m.kind == "surprise_success"
    assert m.delta > 0
    assert m.delta == analytics.delta(True, SUCCESS_XP)


def test_surprise_failure_has_negative_delta():
    m = moment(1, FAILURE_XP, completed=False)
    assert m.kind == "surprise_failure"
    assert m.delta < 0
    assert m.delta == analytics.delta(False, FAILURE_XP)


def test_uses_analytics_for_difficulty_and_band():
    m = moment(1, SUCCESS_XP, completed=True)
    assert m.difficulty == analytics.difficulty(SUCCESS_XP)
    assert m.band == analytics.difficulty_band(m.difficulty, CFG)


def test_cfg_is_respected():
    cfg = AnalyticsConfig(surprise_success_max_xp=0.1, commentary_floor_success_max_xp=0.05)
    assert build_moment(make_event(completed=True), 0.2, "test-model", cfg=cfg) is None
    assert build_moment(make_event(completed=True), 0.2, "test-model") is not None


@pytest.mark.parametrize("xp, completed", [(0.0, True), (SUCCESS_XP, True), (FAILURE_XP, False), (1.0, False)])
def test_significance_is_abs_delta_in_unit_interval(xp, completed):
    m = moment(1, xp, completed)
    assert m.significance == abs(m.delta)
    assert 0.0 <= m.significance <= 1.0


@pytest.mark.parametrize("xp", [-0.01, 1.01, float("nan")])
def test_xp_outside_unit_interval_raises(xp):
    with pytest.raises(ValueError, match="xp must be in"):
        build_moment(make_event(), xp, "test-model")


@pytest.mark.parametrize("t, period, minute, added, label", [
    (0.0, 1, 1, 0, "1"), (59.9, 1, 1, 0, "1"), (60.0, 1, 2, 0, "2"), (2700.0, 1, 45, 1, "45+1"),
    (SECOND_HALF_START, 2, 46, 0, "46"), (SECOND_HALF_START + 2700.0 + 61.0, 2, 90, 2, "90+2"),
])
def test_minute_from_clock(t, period, minute, added, label):
    m = moment(1, SUCCESS_XP, completed=True, t=t, period=period)
    assert (m.minute, m.minute_added, m.minute_label) == (minute, added, label)
    assert m.evidence["m1:1:minute"] == minute
    assert m.evidence["m1:1:minute_added"] == added
    assert m.evidence["m1:1:minute_label"] == label


def test_period_two_needs_second_half_start():
    with pytest.raises(ValueError, match="second_half_start"):
        build_moment(make_event(completed=True, t=3000.0, period=2), SUCCESS_XP, "test-model")


# --- evidence -----------------------------------------------------------------------------------

def test_every_number_like_fact_has_an_evidence_key():
    m = moment(3, SUCCESS_XP, completed=True)
    facts = {
        "xp": m.xp, "delta": m.delta, "difficulty": m.difficulty, "difficulty_band": m.band,
        "completed": m.completed, "minute": m.minute, **m.pass_geometry.model_dump(),
    }
    for name, value in facts.items():
        key = f"{m.event_id}:{name}"
        assert key in m.evidence, key
        assert m.evidence[key] == value


def test_evidence_keys_unique_ordered_and_prefixed():
    m = moment(3, SUCCESS_XP, completed=True)
    keys = list(m.evidence)
    assert len(keys) == len(set(keys))
    assert keys == [f"m1:3:{name}" for name in EVIDENCE_NAMES]


def test_evidence_values_are_raw():
    xp = SUCCESS_XP - 0.0123456789
    m = moment(1, xp, completed=True)
    assert m.evidence["m1:1:xp"] == xp
    assert m.evidence["m1:1:delta"] == 1 - xp
    assert m.evidence["m1:1:length"] == 22.19
    assert m.evidence["m1:1:start_y"] == 22.25


def test_completed_stays_bool_after_json_round_trip():
    m = moment(1, SUCCESS_XP, completed=True)
    back = Moment.model_validate_json(m.model_dump_json())
    assert back == m
    assert type(back.evidence["m1:1:completed"]) is bool
    assert type(back.evidence["m1:1:difficulty"]) is int
    assert json.loads(m.model_dump_json())["evidence"]["m1:1:completed"] is True


def test_same_input_gives_identical_json():
    a = build_moment(make_event(), SUCCESS_XP, "test-model", player_names={"brk_p7": "Alan Ashley"},
                     team_names={"brk": "Brackmoor Athletic"})
    b = build_moment(make_event(), SUCCESS_XP, "test-model", player_names={"brk_p7": "Alan Ashley"},
                     team_names={"brk": "Brackmoor Athletic"})
    assert a.model_dump_json() == b.model_dump_json()


def test_names_in_evidence_only_when_provided():
    bare = moment(1, SUCCESS_XP, completed=True)
    assert bare.player_name is None and bare.team_name is None
    assert not any(k.endswith((":player_name", ":team_name")) for k in bare.evidence)

    named = build_moment(make_event(1), SUCCESS_XP, "test-model",
                         player_names={"brk_p7": "Alan Ashley"}, team_names={"brk": "Brackmoor Athletic"})
    assert named.player_name == "Alan Ashley" and named.team_name == "Brackmoor Athletic"
    assert list(named.evidence)[-2:] == ["m1:1:player_name", "m1:1:team_name"]
    assert named.evidence["m1:1:player_name"] == "Alan Ashley"

    other = build_moment(make_event(1), SUCCESS_XP, "test-model", player_names={"someone_else": "X"})
    assert other.player_name is None and "m1:1:player_name" not in other.evidence


def test_moment_is_frozen_and_rejects_unknown_fields():
    m = moment(1, SUCCESS_XP, completed=True)
    with pytest.raises(ValidationError):
        m.xp = 0.5
    with pytest.raises(ValidationError):
        Moment.model_validate({**m.model_dump(), "surprise": "yes"})


# --- select_highlights / qualifies_live ---------------------------------------------------------

def test_select_highlights_keeps_both_kinds_and_caps():
    failures = [moment(i, 0.99, completed=False) for i in range(10)]          # significance 0.99
    successes = [moment(100 + i, SUCCESS_XP, completed=True) for i in range(2)]  # smaller significance
    top = select_highlights(failures + successes, per_kind=3)
    assert set(top) == {"surprise_success", "surprise_failure"}
    assert len(top["surprise_failure"]) == 3
    assert [m.seq for m in top["surprise_success"]] == [100, 101]
    assert all(m.kind == "surprise_failure" for m in top["surprise_failure"])


def test_select_highlights_sorts_by_significance_then_seq():
    ms = [moment(5, 0.95, False), moment(2, 0.95, False), moment(9, 0.99, False), moment(1, 0.93, False)]
    top = select_highlights(ms, per_kind=3)["surprise_failure"]
    assert [m.seq for m in top] == [9, 2, 5]


def test_select_highlights_zero_and_negative():
    ms = [moment(1, 0.99, False), moment(2, 0.1, True)]
    assert select_highlights(ms, 0) == {"surprise_success": [], "surprise_failure": []}
    assert select_highlights([], 5) == {"surprise_success": [], "surprise_failure": []}
    with pytest.raises(ValueError):
        select_highlights(ms, -1)


def test_qualifies_live_early_then_crowded_out():
    candidate = moment(50, 0.95, completed=False)
    assert qualifies_live(candidate, [], per_kind=2)
    assert qualifies_live(candidate, [moment(1, 0.99, False)], per_kind=2)
    stronger = [moment(1, 0.99, False), moment(2, 0.98, False)]
    assert not qualifies_live(candidate, stronger, per_kind=2)
    # stronger moments of the other kind do not crowd it out
    other_kind = [moment(3, 0.01, True), moment(4, 0.02, True)]
    assert qualifies_live(candidate, other_kind, per_kind=2)


def test_select_half_highlights_ranks_each_half_separately():
    first = [moment(1, 0.99, False, period=1), moment(2, 0.93, False, period=1), moment(3, 0.10, True, period=1)]
    second = [moment(4, 0.95, False, period=2, t=3000.0), moment(5, 0.30, True, period=2, t=3100.0)]
    top = select_half_highlights(first + second, per_kind=1)
    assert set(top) == {1, 2}
    assert [m.seq for m in top[1]["surprise_failure"]] == [1]
    assert [m.seq for m in top[2]["surprise_failure"]] == [4]  # 0.99 in the first half does not count
    assert [m.seq for m in top[2]["surprise_success"]] == [5]
    assert top[1] == select_highlights(first, 1) and top[2] == select_highlights(second, 1)
    assert select_half_highlights([], 3) == {p: {"surprise_success": [], "surprise_failure": []} for p in (1, 2)}
