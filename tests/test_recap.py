import json

import pytest

from xplain import clock
from xplain.pipeline import pass_marks
from xplain.recap import (MINUS, PASSING_BAND, half_up, recap_card, recap_claims, recap_facts, roster,
                          team_passing, vs_expected_label)
from xplain.schema import MatchMetaEvent, ShotEvent

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)


def with_goals(events, goals):
    """seed42 (0-0) with its shots made non-goals and goals added: (side "home"/"away", period, minute)."""
    meta = next(e for e in events if isinstance(e, MatchMetaEvent))
    start = clock.second_half_start(events)
    shot = next(e for e in events if isinstance(e, ShotEvent))
    out = [e.model_copy(update={"goal": False}) if isinstance(e, ShotEvent) else e for e in events]
    for i, (side, period, minute) in enumerate(goals):
        team = meta.home_team_id if side == "home" else meta.away_team_id
        player = next(p for p in meta.players if p.team_id == team and p.player_id.endswith(f"p{9 + i % 2}"))
        t = (minute - 1) * 60.0 + 5 if period == 1 else start + (minute - 46) * 60.0 + 5
        seq = max(e.seq for e in events if e.t <= t)
        out.append(shot.model_copy(update={"event_id": f"goal-{i}", "t": t, "period": period, "team_id": team,
                                           "player_id": player.player_id, "seq": seq, "goal": True,
                                           "on_target": True}))
    return out, meta


@pytest.fixture(scope="module")
def marks(seed42_events, tiny_scorer):  # noqa: F811
    return pass_marks(seed42_events, tiny_scorer, WIDE)


def card(events, marks, scope="full_time"):
    return recap_card(events, marks, scope)


def test_vs_expected_label():
    assert vs_expected_label(19) == "vs expected +19"
    assert vs_expected_label(-19) == f"vs expected {MINUS}19" == "vs expected −19"
    assert vs_expected_label(0) == "vs expected 0"


def test_team_passing_is_computed_from_the_marks(seed42_events, marks):  # noqa: F811
    meta = next(e for e in seed42_events if isinstance(e, MatchMetaEvent))
    t = team_passing(marks, meta.home_team_id, meta.home_name, "full_time")
    mine = [m for m in marks if m.team_id == meta.home_team_id]
    completed, expected = sum(m.completed for m in mine), half_up(sum(m.xp for m in mine))
    assert (t.passes, t.completed, t.expected) == (len(mine), completed, expected)
    assert t.vs_expected == completed - expected and t.label == vs_expected_label(t.vs_expected)
    gap = t.vs_expected
    assert t.band == ("above" if gap >= PASSING_BAND["full_time"] else "below"
                      if gap <= -PASSING_BAND["full_time"] else "close")
    half = team_passing(marks, meta.home_team_id, meta.home_name, "half_time")
    assert half.passes == sum(1 for m in mine if m.period == 1)


def test_goalless_draw(seed42_events, marks):  # noqa: F811
    c = card(seed42_events, marks)
    assert (c.score_home, c.score_away, c.goals, c.result) == (0, 0, [], "draw")
    assert c.comeback_team_id is None and c.late_winner_team_id is None and c.late_equaliser_team_id is None
    assert "without a goal" in recap_claims(c)["result"]
    assert card(seed42_events, marks, "half_time").result == "level"


def test_comeback_with_a_late_winner(seed42_events, marks):  # noqa: F811
    events, meta = with_goals(seed42_events, [("away", 1, 23), ("home", 2, 83), ("home", 2, 90)])
    c = card(events, marks)
    assert (c.score_home, c.score_away, c.result) == (2, 1, "home_win")
    assert [(g.minute_label, g.late, g.home, g.away) for g in c.goals] == [
        ("23", False, 0, 1), ("83", False, 1, 1), ("90", True, 2, 1)]
    assert c.comeback_team_id == meta.home_team_id and c.late_winner_team_id == meta.home_team_id
    assert c.late_winner_scorer == c.goals[-1].player_name
    claims = recap_claims(c)
    assert {f"comeback:{meta.home_name}", f"late_winner:{meta.home_name}", "result"} <= set(claims)


def test_scorer_claims_one_per_player(seed42_events, marks):  # noqa: F811
    events, _ = with_goals(seed42_events, [("away", 1, 23), ("home", 2, 83), ("home", 2, 90)])
    c = card(events, marks)
    assert {k for k in recap_claims(c) if k.startswith("scorer:")} == {f"scorer:{g.player_name}" for g in c.goals}


def test_an_early_winner_is_not_late_and_half_time_has_no_late_claims(seed42_events, marks):  # noqa: F811
    events, meta = with_goals(seed42_events, [("home", 1, 10), ("home", 2, 88)])
    c = card(events, marks)
    assert c.result == "home_win" and c.late_winner_team_id is None and c.comeback_team_id is None
    assert c.late_winner_scorer is None
    ht = card(events, marks, "half_time")
    assert (ht.score_home, ht.score_away, ht.result) == (1, 0, "home_lead") and len(ht.goals) == 1
    assert not any(k.startswith(("late_", "comeback")) for k in recap_claims(ht))


def test_late_equaliser(seed42_events, marks):  # noqa: F811
    events, meta = with_goals(seed42_events, [("home", 1, 30), ("away", 2, 90)])
    c = card(events, marks)
    assert c.result == "draw" and c.late_equaliser_team_id == meta.away_team_id
    assert c.late_equaliser_scorer == c.goals[-1].player_name
    assert f"late_equaliser:{meta.away_name}" in recap_claims(c)


def test_facts_carry_no_numbers(seed42_events, marks):  # noqa: F811
    events, meta = with_goals(seed42_events, [("away", 1, 23), ("home", 2, 83), ("home", 2, 90)])
    facts = recap_facts(card(events, marks))
    text = json.dumps(facts, ensure_ascii=False)
    assert not any(ch.isdigit() for ch in text)
    no_slots = json.dumps({k: v for k, v in facts.items() if k != "slots"}, ensure_ascii=False)
    assert "minute" not in text and "expected\":" not in no_slots and "vs_expected" not in no_slots
    assert facts["margin"] == "narrow" and facts["passing"].keys() == {meta.home_name, meta.away_name}
    assert [g["late"] for g in facts["goals"]] == [False, False, True]


def test_roster_and_bad_scope(seed42_events, marks):  # noqa: F811
    assert len(roster(seed42_events)) == 22
    with pytest.raises(ValueError):
        card(seed42_events, marks, "extra_time")
