"""
Data contracts passed between agents.

A ``Moment`` is one flagged pass (a surprise success or a surprise failure) with everything the
later agents may talk about. ``evidence`` holds every narratable fact under a stable key
``"{event_id}:{name}"``, with raw unrounded values; rounding rules belong to the Verifier.
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
