"""
Data contracts passed between agents.

A ``Moment`` is one flagged pass (a surprise success or a surprise failure) with everything the
later agents may talk about. ``evidence`` holds every narratable fact under a stable key
``"{event_id}:{name}"``, with raw unrounded values; rounding rules belong to the Verifier.
An ``Explanation`` holds up to three SHAP drivers of one Moment, one per pass input (a one-hot
group counts as one input). SHAP values are in log-odds of the model's raw (uncalibrated) score:
they are drivers of that score, never percentage points of xP.
Same strict style as schema.py: unknown fields are rejected and instances are immutable.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .analytics import Band, Outlier
from .schema import X, Y

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PassGeometry(_Strict):
    """Where the pass went and how fast. Same names and meaning as PassInputs/PassDetails; display only."""

    start_x: X
    start_y: Y
    end_x: X
    end_y: Y
    length: Annotated[float, Field(ge=0.0)]
    ball_speed_ms: Annotated[float, Field(ge=0.0, le=45.0)]


class Moment(_Strict):
    """One flagged pass, ready for the Explainer, Narrator and Verifier."""

    moment_id: str  # f"moment:{event_id}"
    kind: Outlier
    event_id: str
    match_id: str
    seq: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0.0)]
    minute: Annotated[int, Field(ge=1)]
    period: Literal[1, 2]
    team_id: str
    team_name: str | None = None
    player_id: str | None
    player_name: str | None = None
    completed: bool
    xp: Probability
    delta: Annotated[float, Field(ge=-1.0, le=1.0)]
    difficulty: Annotated[int, Field(ge=0, le=100)]
    band: Band
    significance: Probability  # = abs(delta)
    pass_geometry: PassGeometry
    evidence: dict[str, bool | int | float | str]
    model_version: str


class Driver(_Strict):
    """One of the top input units behind the model's raw score for a pass.

    A unit is one pass input: pitch position (the four location columns), a one-hot group (height,
    body part, play pattern) or under_pressure. Its shap is the sum of its columns.
    """

    rank: Annotated[int, Field(ge=1, le=3)]
    feature: str           # "position", "height", "body_part", "play_pattern" or "under_pressure"
    topic: str             # what the driver is about, e.g. "pass height"
    value: float | str     # what the pass had: "attacking_third|out_wide", "Head", 1.0 for under pressure
    label: str             # short phrase the Narrator may use, e.g. "with a header"; never contains digits
    shap: float            # log-odds (sum over the unit's columns), display only
    direction: Literal["raises", "lowers"]  # "raises": pushes the raw score toward completion (shap > 0)


class Explanation(_Strict):
    """SHAP explanation of one Moment: baseline + driver shap values + rest = raw_score."""

    moment_id: str
    event_id: str
    method: str            # "xgboost pred_contribs, mean of 5 fold models"
    unit: str              # "log-odds of the raw model score, not percentage points of xP"
    baseline: float
    raw_score: float
    drivers: Annotated[list[Driver], Field(min_length=0, max_length=3)]  # 0 to 3, never padded
    rest: float            # sum of the shap of every unit that is not a driver
    evidence: dict[str, bool | int | float | str]
    model_version: str
