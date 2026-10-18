"""
Recap: the full-time (or half-time) recap card and the facts the RecapEditor model may write about.

Every number on the card is computed here, in code: the score, each goal's minute_label (clock.match_clock), and
per team its passes, completed passes, expected completions (the sum of xP) and "vs expected" (completed minus
expected, rounded to a whole pass), with its label ("vs expected +19"). The model never sees a number: recap_facts
gives it categories only (the result, the margin as "narrow" or "clear", each goal's scorer, half and whether it
was late, each team's passing band) and the claims it may make (recap_claims). The RecapEditor writes one short
recap per language (RecapOut) and lists the claims it used; recap_verifier checks it against the card.

Editorial constants (change them here only):
  * LATE_MINUTE: a goal is "late" from this base minute of its half on, added time included (85' in the second
    half, 40' in the first)
  * PASSING_BAND: a team is "above" or "below" expected when |vs expected| is at least this many passes, else
    "close". Chosen from seeds 3001, 42 and 7 (full time: +19.4 / -19.1, +16.8 / -10.4, -0.7 / -1.0; halves
    between -13.4 and +16.0), so a clear gap reads as one and a near-zero one does not.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import clock
from .contracts import PassMark
from .schema import MatchMetaEvent, ShotEvent

RecapScope = Literal["full_time", "half_time"]
RECAP_SCOPES: tuple[str, ...] = ("full_time", "half_time")
SCOPE_PERIODS: dict[str, tuple[int, ...]] = {"full_time": (1, 2), "half_time": (1,)}
Result = Literal["home_win", "away_win", "draw", "home_lead", "away_lead", "level"]
Band = Literal["above", "below", "close"]

LATE_MINUTE: dict[int, int] = {1: 40, 2: 85}
PASSING_BAND: dict[str, float] = {"full_time": 8.0, "half_time": 5.0}
MINUS = "−"   # the Studio's minus sign ("−94%")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RecapOut(_Strict):
    """What the RecapEditor returns: the recap per language and the ids of the claims it makes (recap_claims)."""

    text_en: str
    text_es: str
    text_el: str
    claims_used: list[str]


class RecapGoal(_Strict):
    team_id: str
    team_name: str
    player_name: str | None = None
    minute_label: str
    period: Literal[1, 2]
    late: bool
    home: Annotated[int, Field(ge=0)]   # the score after the goal
    away: Annotated[int, Field(ge=0)]


class TeamPassing(_Strict):
    team_id: str
    name: str
    passes: Annotated[int, Field(ge=0)]
    completed: Annotated[int, Field(ge=0)]
    expected: float              # the sum of xP over the team's passes, to one decimal
    vs_expected: int             # completed - expected, rounded to a whole pass
    label: str                   # "vs expected +19", "vs expected −19", "vs expected 0"
    band: Band


class RecapCard(_Strict):
    """The recap card of one scope. Every number here is computed in code."""

    scope: RecapScope
    match_id: str
    home: TeamPassing
    away: TeamPassing
    score_home: Annotated[int, Field(ge=0)]
    score_away: Annotated[int, Field(ge=0)]
    goals: list[RecapGoal]
    result: Result
    comeback_team_id: str | None = None        # the winner (or leader) was behind at some point
    late_winner_team_id: str | None = None     # full time: the goal that put the winner ahead for good was late
    late_equaliser_team_id: str | None = None  # full time: a draw whose last goal levelled it late


def vs_expected_label(value: int) -> str:
    if value > 0:
        return f"vs expected +{value}"
    if value < 0:
        return f"vs expected {MINUS}{-value}"
    return "vs expected 0"


def team_passing(marks: Sequence[PassMark], team_id: str, name: str, scope: str) -> TeamPassing:
    """A team's passing in the scope: counts, the sum of xP and completed minus expected, with its band."""
    ps = [m for m in marks if m.team_id == team_id and m.period in SCOPE_PERIODS[scope]]
    completed = sum(m.completed for m in ps)
    expected = sum(m.xp for m in ps)
    gap = completed - expected
    band: Band = "above" if gap >= PASSING_BAND[scope] else "below" if gap <= -PASSING_BAND[scope] else "close"
    value = round(gap)
    return TeamPassing(team_id=team_id, name=name, passes=len(ps), completed=completed, expected=round(expected, 1),
                       vs_expected=value, label=vs_expected_label(value), band=band)


def _result(home: int, away: int, scope: str) -> Result:
    if scope == "full_time":
        return "home_win" if home > away else "away_win" if away > home else "draw"
    return "home_lead" if home > away else "away_lead" if away > home else "level"


def recap_card(events: Sequence, marks: Sequence[PassMark], scope: str) -> RecapCard:
    """The card for full time or half time, from the match events and every pass's PassMark."""
    meta = next((e for e in events if isinstance(e, MatchMetaEvent)), None)
    if meta is None:
        raise ValueError("a recap needs the match_meta event")
    if scope not in RECAP_SCOPES:
        raise ValueError(f"scope must be one of {RECAP_SCOPES}, got {scope!r}")
    periods = SCOPE_PERIODS[scope]
    start = clock.second_half_start(events)
    names = {p.player_id: p.name for p in meta.players}
    teams = {meta.home_team_id: meta.home_name, meta.away_team_id: meta.away_name}
    goals: list[RecapGoal] = []
    home = away = 0
    for shot in sorted((e for e in events if isinstance(e, ShotEvent) and e.goal and e.period in periods),
                       key=lambda e: e.seq):
        if shot.team_id == meta.home_team_id:
            home += 1
        else:
            away += 1
        c = clock.match_clock(shot.t, shot.period, start)
        goals.append(RecapGoal(team_id=shot.team_id, team_name=teams[shot.team_id],
                               player_name=names.get(shot.player_id) if shot.player_id else None,
                               minute_label=c.label, period=shot.period,
                               late=c.minute >= LATE_MINUTE[shot.period], home=home, away=away))
    result = _result(home, away, scope)
    ahead = meta.home_team_id if home > away else meta.away_team_id if away > home else None
    comeback = late_winner = late_equaliser = None
    if ahead is not None:
        behind_once = any((g.home < g.away) if ahead == meta.home_team_id else (g.away < g.home) for g in goals)
        comeback = ahead if behind_once else None
        if scope == "full_time":
            # the goal after which the winner led for good: the last goal that took them from level to ahead
            decisive = next(g for g in reversed(goals)
                            if g.team_id == ahead and abs(g.home - g.away) == 1
                            and all((x.home != x.away) for x in goals[goals.index(g):]))
            late_winner = ahead if decisive.late else None
    elif scope == "full_time" and goals and goals[-1].late:
        late_equaliser = goals[-1].team_id
    return RecapCard(scope=scope, match_id=meta.match_id,
                     home=team_passing(marks, meta.home_team_id, meta.home_name, scope),
                     away=team_passing(marks, meta.away_team_id, meta.away_name, scope),
                     score_home=home, score_away=away, goals=goals, result=result, comeback_team_id=comeback,
                     late_winner_team_id=late_winner, late_equaliser_team_id=late_equaliser)


def roster(events: Sequence) -> list[str]:
    """Every player name of the match (for recap_verifier: a recap names no player it has no claim about)."""
    meta = next((e for e in events if isinstance(e, MatchMetaEvent)), None)
    return [p.name for p in meta.players] if meta else []


# --- what the model sees -----------------------------------------------------------------------------------------

def _team_name(card: RecapCard, team_id: str | None) -> str | None:
    return {card.home.team_id: card.home.name, card.away.team_id: card.away.name}.get(team_id or "")


def recap_claims(card: RecapCard) -> dict[str, str]:
    """The claims the recap may make, by id, each with what it says (English, for the model). No numbers: ids use
    names, not player or team ids."""
    home, away = card.home.name, card.away.name
    result = {
        "home_win": f"{home} won the match against {away}",
        "away_win": f"{away} won the match against {home}",
        "draw": f"{home} and {away} drew" + (" without a goal" if not card.goals else ""),
        "home_lead": f"{home} lead {away} at half time",
        "away_lead": f"{away} lead {home} at half time",
        "level": f"{home} and {away} are level at half time" + (" without a goal" if not card.goals else ""),
    }[card.result]
    claims = {"result": result}
    if (team := _team_name(card, card.comeback_team_id)) is not None:
        claims[f"comeback:{team}"] = f"{team} came from behind"
    if (team := _team_name(card, card.late_winner_team_id)) is not None:
        claims[f"late_winner:{team}"] = f"{team} scored the winning goal late in the match"
    if (team := _team_name(card, card.late_equaliser_team_id)) is not None:
        claims[f"late_equaliser:{team}"] = f"{team} levelled the match late on"
    for g in card.goals:
        if g.player_name and f"scorer:{g.player_name}" not in claims:
            claims[f"scorer:{g.player_name}"] = f"{g.player_name} scored for {g.team_name}"
    band_text = {"above": "completed more passes than expected", "below": "completed fewer passes than expected",
                 "close": "completed about as many passes as expected"}
    for t in (card.home, card.away):
        claims[f"passing:{t.name}"] = f"{t.name} {band_text[t.band]}"
    return claims


def recap_facts(card: RecapCard) -> dict:
    """The facts the RecapEditor may write about: categories and names only, no numbers and no minutes."""
    margin = abs(card.score_home - card.score_away)
    return {
        "scope": card.scope,
        "home": card.home.name,
        "away": card.away.name,
        "result": card.result,
        "margin": None if margin == 0 else "narrow" if margin == 1 else "clear",
        "goalless": not card.goals,
        "goals": [{"team": g.team_name, "scorer": g.player_name, "half": "first" if g.period == 1 else "second",
                   "late": g.late} for g in card.goals],
        "passing": {card.home.name: card.home.band, card.away.name: card.away.band},
        "claims": recap_claims(card),
    }
