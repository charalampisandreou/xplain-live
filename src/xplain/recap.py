"""
Recap: the full-time (or half-time) recap card and the facts the RecapEditor model may write about.

Every number on the card is computed here, in code: the score, each goal's minute_label (clock.match_clock), and
per team its passes, completed passes, expected completions (the sum of xP, rounded half-up to a whole pass) and
"vs expected" (completed minus that expected, so the numbers on screen always add up), with its label
("vs expected +19"); and the star player of the scope (star_player). The model never sees a number: recap_facts
gives it categories only (the result, the margin as "narrow" or "clear", each goal's scorer, half and whether it
was late, each team's passing band, the star's name) and the claims it may make (recap_claims).

Recap v2: the model may write four slots, never a number itself: {home_vs_expected}, {away_vs_expected},
{star_name}, {star_pva} (SLOTS). Code fills them from the card (slot_values) after the raw text has passed
recap_verifier: the size of a team's gap (no sign; the text says "more" or "fewer"), the star's name and how many
more passes than expected the star completed. A team's slot exists only when its band is above or below; the star
slots only when the card has a star. Completed, expected and passes stay on the card, under fixed labels.

On screen, Expected and vs expected show two decimals: expected_xp (the sum of xP) and vs_expected_xp (completed
minus that sum), each rounded by round2, the same rounding as the page's toFixed(2) (player.js fixed2). The whole
numbers above stay for the text's slots, the bands and the star rule; the text's number is always the on-screen
value rounded to a whole pass (|vs_expected - vs_expected_xp| <= 0.5).

Editorial constants (change them here only):
  * LATE_MINUTE: a goal is "late" from this base minute of its half on, added time included (85' in the second
    half, 40' in the first)
  * PASSING_BAND: a team is "above" or "below" expected when |vs expected| is at least this many passes, else
    "close". Chosen from seeds 3001, 42 and 7 (full time: +19.4 / -19.1, +16.8 / -10.4, -0.7 / -1.0; halves
    between -13.4 and +16.0), so a clear gap reads as one and a near-zero one does not.
  * STAR_MIN_PVA: the star must have completed at least this many passes more than expected (after rounding), so
    the text always reads "passes" in the plural and the star is a clear, positive standout
"""
from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
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
PASSING_BAND: dict[str, int] = {"full_time": 8, "half_time": 5}
STAR_MIN_PVA = 2
SLOTS: tuple[str, ...] = ("home_vs_expected", "away_vs_expected", "star_name", "star_pva")
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
    expected: int                # the sum of xP over the team's passes, rounded half-up
    vs_expected: int             # completed - expected
    label: str                   # "vs expected +19", "vs expected −19", "vs expected 0"
    band: Band
    expected_xp: float | None = None      # the sum of xP, round2 (two decimals, for the screen)
    vs_expected_xp: float | None = None   # completed - the sum of xP, round2


class StarPlayer(_Strict):
    """The scope's star: the highest PVA total (completed - expected), at least STAR_MIN_PVA."""

    player_id: str
    name: str
    team_id: str
    team_name: str
    passes: Annotated[int, Field(ge=1)]
    completed: Annotated[int, Field(ge=0)]
    expected: int                # the sum of xP over the player's passes, rounded half-up
    pva: Annotated[int, Field(ge=STAR_MIN_PVA)]   # completed - expected
    expected_xp: float | None = None      # the sum of xP, round2 (two decimals, for the screen)
    vs_expected_xp: float | None = None   # completed - the sum of xP, round2


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
    late_winner_scorer: str | None = None      # who scored that late winner / late equaliser (None: unnamed)
    late_equaliser_scorer: str | None = None
    star: StarPlayer | None = None


def vs_expected_label(value: int) -> str:
    if value > 0:
        return f"vs expected +{value}"
    if value < 0:
        return f"vs expected {MINUS}{-value}"
    return "vs expected 0"


def half_up(x: float) -> int:
    """x rounded to a whole number, halves away from zero (Python's round() rounds halves to even)."""
    return int(Decimal(repr(x)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def xp_sum(marks: Sequence[PassMark]) -> float:
    """The sum of xP added one pass at a time, left to right, as player.js adds it. Python's sum() compensates for
    float rounding (since 3.12), so it can differ from the page in the last bits."""
    total = 0.0
    for m in marks:
        total += m.xp
    return total


def round2(x: float) -> float:
    """x to two decimals the way JavaScript's toFixed(2) does it: on the exact binary value of x, halves away from
    zero (1.005 is stored just below 1.005, so it gives 1.0, as toFixed does). -0.0 becomes 0.0."""
    return float(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)) + 0.0


def team_passing(marks: Sequence[PassMark], team_id: str, name: str, scope: str) -> TeamPassing:
    """A team's passing in the scope: counts, expected (the sum of xP, rounded half-up) and completed minus
    expected, with its band."""
    ps = [m for m in marks if m.team_id == team_id and m.period in SCOPE_PERIODS[scope]]
    completed = sum(m.completed for m in ps)
    xp = xp_sum(ps)
    expected = half_up(xp)
    value = completed - expected
    band: Band = "above" if value >= PASSING_BAND[scope] else "below" if value <= -PASSING_BAND[scope] else "close"
    return TeamPassing(team_id=team_id, name=name, passes=len(ps), completed=completed, expected=expected,
                       vs_expected=value, label=vs_expected_label(value), band=band,
                       expected_xp=round2(xp), vs_expected_xp=round2(completed - xp))


def star_player(marks: Sequence[PassMark], scope: str) -> StarPlayer | None:
    """The player with the highest PVA total (completed - sum of xP) in the scope, both teams; ties go to more
    passes, then the earlier first pass. None when that player's rounded PVA is below STAR_MIN_PVA."""
    by_player: dict[str, list[PassMark]] = {}
    for m in sorted(marks, key=lambda m: m.seq):
        if m.period in SCOPE_PERIODS[scope] and m.player_id:
            by_player.setdefault(m.player_id, []).append(m)
    if not by_player:
        return None
    ps = max(by_player.values(),
             key=lambda ps: (sum(m.completed for m in ps) - sum(m.xp for m in ps), len(ps), -ps[0].seq))
    completed, xp = sum(m.completed for m in ps), xp_sum(ps)   # ps in seq order, as Players adds them
    expected = half_up(xp)
    if completed - expected < STAR_MIN_PVA:
        return None
    first = ps[0]
    return StarPlayer(player_id=first.player_id, name=first.player_name or first.player_id, team_id=first.team_id,
                      team_name=first.team_name or first.team_id, passes=len(ps), completed=completed,
                      expected=expected, pva=completed - expected,
                      expected_xp=round2(xp), vs_expected_xp=round2(completed - xp))


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
    comeback = late_winner = late_equaliser = late_winner_scorer = late_equaliser_scorer = None
    if ahead is not None:
        behind_once = any((g.home < g.away) if ahead == meta.home_team_id else (g.away < g.home) for g in goals)
        comeback = ahead if behind_once else None
        if scope == "full_time":
            # the goal after which the winner led for good: the last goal that took them from level to ahead
            decisive = next(g for g in reversed(goals)
                            if g.team_id == ahead and abs(g.home - g.away) == 1
                            and all((x.home != x.away) for x in goals[goals.index(g):]))
            late_winner = ahead if decisive.late else None
            late_winner_scorer = decisive.player_name if decisive.late else None
    elif scope == "full_time" and goals and goals[-1].late:
        late_equaliser = goals[-1].team_id
        late_equaliser_scorer = goals[-1].player_name
    return RecapCard(scope=scope, match_id=meta.match_id,
                     home=team_passing(marks, meta.home_team_id, meta.home_name, scope),
                     away=team_passing(marks, meta.away_team_id, meta.away_name, scope),
                     score_home=home, score_away=away, goals=goals, result=result, comeback_team_id=comeback,
                     late_winner_team_id=late_winner, late_equaliser_team_id=late_equaliser,
                     late_winner_scorer=late_winner_scorer, late_equaliser_scorer=late_equaliser_scorer,
                     star=star_player(marks, scope))


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
        claims[f"late_winner:{team}"] = f"{team} scored the winning goal late in the match" + (
            f", through {card.late_winner_scorer}" if card.late_winner_scorer else "")
    if (team := _team_name(card, card.late_equaliser_team_id)) is not None:
        claims[f"late_equaliser:{team}"] = f"{team} levelled the match late on" + (
            f", through {card.late_equaliser_scorer}" if card.late_equaliser_scorer else "")
    for g in card.goals:
        if g.player_name and f"scorer:{g.player_name}" not in claims:
            claims[f"scorer:{g.player_name}"] = f"{g.player_name} scored for {g.team_name}"
    band_text = {"above": "completed more passes than expected", "below": "completed fewer passes than expected",
                 "close": "completed about as many passes as expected"}
    for t in (card.home, card.away):
        claims[f"passing:{t.name}"] = f"{t.name} {band_text[t.band]}"
    if card.star is not None:
        where = "in the match" if card.scope == "full_time" else "in the first half"
        claims[f"star:{card.star.name}"] = (f"{card.star.name} ({card.star.team_name}) completed the most passes "
                                             f"above expected {where}")
    return claims


def slot_values(card: RecapCard) -> dict[str, str]:
    """The slots this card has, with the text code puts in their place (the same in EN, ES and EL: whole numbers
    and a name). A team's slot only when its band is above or below; the star's only when there is a star."""
    values: dict[str, str] = {}
    for side, t in (("home", card.home), ("away", card.away)):
        if t.band != "close":
            values[f"{side}_vs_expected"] = str(abs(t.vs_expected))
    if card.star is not None:
        values["star_name"] = card.star.name
        values["star_pva"] = str(card.star.pva)
    return values


def slot_directions(card: RecapCard) -> dict[str, str]:
    """The direction each number slot must be written with: "more" (above, and the star) or "fewer" (below)."""
    out = {f"{side}_vs_expected": ("more" if t.band == "above" else "fewer")
           for side, t in (("home", card.home), ("away", card.away)) if t.band != "close"}
    if card.star is not None:
        out["star_pva"] = "more"
    return out


def slot_team(card: RecapCard) -> dict[str, str]:
    """The team name each team slot belongs to."""
    return {"home_vs_expected": card.home.name, "away_vs_expected": card.away.name}


def _slot_meanings(card: RecapCard) -> dict[str, str]:
    """What each available slot stands for (for the model; no numbers)."""
    meanings = {}
    for side, t in (("home", card.home), ("away", card.away)):
        if t.band != "close":
            word = "more" if t.band == "above" else "fewer"
            meanings[f"{side}_vs_expected"] = (f"how many {word} passes than expected {t.name} completed; write it "
                                               f"right before the '{word} ... than expected' phrase, in a clause "
                                               f"naming {t.name}")
    if card.star is not None:
        meanings["star_name"] = "the star's name, in place of writing it"
        meanings["star_pva"] = ("how many more passes than expected the star completed; write it right before the "
                                "'more ... than expected' phrase, in the clause with {star_name}")
    return meanings


def star_facts(card: RecapCard) -> dict:
    """What the model may say about the star besides {star_name} and {star_pva}: the team, whether the star scored
    in the scope, and whether that was the late winner or late equaliser (categories, no numbers)."""
    star = card.star
    scored = any(g.player_name == star.name for g in card.goals)
    return {"team": star.team_name, "scored": scored,
            "scored_late_winner": card.late_winner_scorer == star.name,
            "scored_late_equaliser": card.late_equaliser_scorer == star.name}


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
        "star": None if card.star is None else star_facts(card),
        "claims": recap_claims(card),
        "slots": _slot_meanings(card),
    }
