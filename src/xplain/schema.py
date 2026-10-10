"""
Event schema for the synthetic match feed.

Pass events keep two groups of fields strictly apart:

* ``PassInputs``: information available at the moment the pass is played. These are the ONLY
  fields the scoring model may see.
* ``PassDetails``: descriptive and outcome fields (end location, speed, distance, result).
  They are for display, statistics and evaluation, and are never passed to the model.

The feature builder accepts ``PassInputs`` only, so an end-derived field cannot reach the model.
Coordinates use the StatsBomb pitch: 120 x 80, the attacking goal centre at (120, 40).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

X = Annotated[float, Field(ge=0.0, le=PITCH_LENGTH)]
Y = Annotated[float, Field(ge=0.0, le=PITCH_WIDTH)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PassHeight(StrEnum):
    GROUND = "Ground Pass"
    LOW = "Low Pass"
    HIGH = "High Pass"


class BodyPart(StrEnum):
    FOOT = "Foot"
    HEAD = "Head"
    KEEPER_ARM = "Keeper Arm"
    OTHER = "Other"


class PlayPattern(StrEnum):
    REGULAR = "Regular Play"
    KICK_OFF = "Kick Off"
    THROW_IN = "Throw In"
    FREE_KICK = "Free Kick"
    GOAL_KICK = "Goal Kick"
    KEEPER = "Keeper"
    CORNER = "Corner"
    OTHER = "Other"


class PassInputs(_Strict):
    """Model-input fields: known when the pass is played."""

    start_x: X
    start_y: Y
    height: PassHeight
    body_part: BodyPart
    play_pattern: PlayPattern
    under_pressure: bool


class PassDetails(_Strict):
    """Descriptive and outcome fields: never fed to the model."""

    end_x: X
    end_y: Y
    length: Annotated[float, Field(ge=0.0)]
    ball_speed_ms: Annotated[float, Field(ge=0.0, le=45.0)]
    completed: bool
    recipient_id: str | None = None


class EventBase(_Strict):
    event_id: str
    match_id: str
    seq: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0.0, description="Match clock in seconds since kick-off")]
    period: Literal[1, 2]
    team_id: str
    player_id: str | None = None


class PassEvent(EventBase):
    type: Literal["pass"] = "pass"
    inputs: PassInputs
    details: PassDetails


class ShotEvent(EventBase):
    type: Literal["shot"] = "shot"
    x: X
    y: Y
    shot_speed_ms: Annotated[float, Field(ge=0.0, le=45.0)]
    on_target: bool
    goal: bool


class TackleEvent(EventBase):
    type: Literal["tackle"] = "tackle"
    x: X
    y: Y
    won: bool


class PossessionChangeEvent(EventBase):
    type: Literal["possession_change"] = "possession_change"
    x: X
    y: Y
    gained_by_team_id: str


class PressureEvent(EventBase):
    type: Literal["pressure"] = "pressure"
    x: X
    y: Y
    pressed_player_id: str


class PlayerInfo(_Strict):
    player_id: str
    name: str
    position: Literal["GK", "DEF", "MID", "FWD"]
    shirt_number: Annotated[int, Field(ge=1, le=99)]
    team_id: str


class MatchMetaEvent(EventBase):
    type: Literal["match_meta"] = "match_meta"
    home_team_id: str
    away_team_id: str
    home_name: str
    away_name: str
    seed: int
    generator_version: str
    players: Annotated[list[PlayerInfo], Field(min_length=22, max_length=22)]


class PlayerLoadEvent(EventBase):
    """Physical load so far in the match. All three values are cumulative or running maxima."""

    type: Literal["player_load"] = "player_load"
    player_id: str
    distance_m: Annotated[float, Field(ge=0.0)]
    top_speed_ms: Annotated[float, Field(ge=0.0, le=12.0)]
    sprint_count: Annotated[int, Field(ge=0)]


Event = Annotated[
    PassEvent
    | ShotEvent
    | TackleEvent
    | PossessionChangeEvent
    | PressureEvent
    | MatchMetaEvent
    | PlayerLoadEvent,
    Field(discriminator="type"),
]
