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

from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


QualifiedBy = Literal["ranking", "threshold"]
QUALIFIED_BY_ORDER: tuple[str, ...] = get_args(QualifiedBy)
REPORT_PERIOD: dict[str, int] = {"first_half": 1, "second_half": 2}


class MomentBundle(_Strict):
    """A Moment, its Explanation, and whether (and why) it was commentated when it happened.

    qualified_by: "ranking" when the moment entered the running top K of its kind in its half
    (counting only earlier moments of the same period), "threshold" when it cleared the
    commentary floor. Ordered ("ranking", "threshold"); empty when not commentated.
    """

    moment: Moment
    explanation: Explanation
    qualifies_live: bool
    qualified_by: tuple[QualifiedBy, ...]

    @model_validator(mode="after")
    def _check_qualified(self) -> MomentBundle:
        expected = tuple(r for r in QUALIFIED_BY_ORDER if r in self.qualified_by)
        if self.qualified_by != expected:
            raise ValueError(f"qualified_by must be unique and ordered {QUALIFIED_BY_ORDER}, got {self.qualified_by}")
        if self.qualifies_live != bool(self.qualified_by):
            raise ValueError(f"qualifies_live ({self.qualifies_live}) must equal bool(qualified_by) ({self.qualified_by})")
        return self


class Report(_Strict):
    """Final ranking of one half: the top K MomentBundles of each kind among that half's moments.

    Both kinds are always keys (an empty list when there are none). Each list is sorted by
    (-significance, seq) and holds only moments of the report's period; the cap K
    (highlights_per_kind) is applied by the pipeline.
    """

    scope: Literal["first_half", "second_half"]
    highlights: dict[Outlier, list[MomentBundle]]

    @model_validator(mode="after")
    def _check_highlights(self) -> Report:
        if set(self.highlights) != set(get_args(Outlier)):
            raise ValueError(f"highlights must have exactly the keys {get_args(Outlier)}, got {list(self.highlights)}")
        period = REPORT_PERIOD[self.scope]
        for kind, bundles in self.highlights.items():
            if any(b.moment.kind != kind for b in bundles):
                raise ValueError(f"highlights[{kind!r}] holds a moment of another kind")
            if any(b.moment.period != period for b in bundles):
                raise ValueError(f"{self.scope} report holds a moment outside period {period}")
            keys = [(-b.moment.significance, b.moment.seq) for b in bundles]
            if keys != sorted(keys):
                raise ValueError(f"highlights[{kind!r}] is not sorted by (-significance, seq)")
        return self


class TimelineItem(_Strict):
    """A commentated moment and where it stands, at a given time, in the ranking of its kind and half.

    rank is its 1-based place among all moments of its kind and period so far; it is None exactly
    when the moment is outside the top K (in_ranking False): displaced, or commentated via the floor only.
    """

    bundle: MomentBundle
    in_ranking: bool
    rank: Annotated[int, Field(ge=1)] | None

    @model_validator(mode="after")
    def _check_rank(self) -> TimelineItem:
        if (self.rank is None) == self.in_ranking:
            raise ValueError(f"rank must be set exactly when in_ranking is True (in_ranking={self.in_ranking}, "
                             f"rank={self.rank})")
        return self


class PassMark(_Strict):
    """Light record of ONE pass (every pass, not only outliers), for pass vectors, hover details and pass maps.

    Values are raw and unrounded; rounding is the UI's job. No SHAP and no evidence.
    """

    event_id: str
    seq: Annotated[int, Field(ge=0)]
    t: Annotated[float, Field(ge=0.0)]
    minute: Annotated[int, Field(ge=1)]   # floor(t / 60) + 1, same convention as Moment
    period: Literal[1, 2]
    team_id: str
    team_name: str | None = None
    player_id: str | None = None
    player_name: str | None = None
    pass_geometry: PassGeometry
    completed: bool
    xp: Probability
    delta: Annotated[float, Field(ge=-1.0, le=1.0)]  # outcome - xP (1 completed, 0 failed): the pass's PVA
    difficulty: Annotated[int, Field(ge=0, le=100)]
    band: Band
    outlier: Outlier | None = None   # the outlier kind, when the pass is an outlier
    moment_id: str | None = None     # the Moment's id, when the pass is an outlier

    @model_validator(mode="after")
    def _check_outlier_link(self) -> PassMark:
        if (self.outlier is None) != (self.moment_id is None):
            raise ValueError(f"outlier and moment_id must both be set or both be None "
                             f"(outlier={self.outlier}, moment_id={self.moment_id})")
        return self


class PlayerSummary(_Strict):
    """One player's passing so far: counts, completion, mean xP and PVA (sum of outcome - xP)."""

    player_id: str
    player_name: str | None = None
    team_id: str
    passes: Annotated[int, Field(ge=1)]
    completed: Annotated[int, Field(ge=0)]
    completion_rate: Probability
    mean_xp: Probability
    pva_total: float              # sum of delta over the player's passes
    pva_per_pass: float           # pva_total / passes
    outliers: Annotated[int, Field(ge=0)]   # how many of the player's passes are outliers
