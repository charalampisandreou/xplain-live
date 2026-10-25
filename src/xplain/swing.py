"""
Analyst mode's numbers: a team's running vs expected through the match, and its passing per difficulty tier.
Pure functions over the bundle's marks, no I/O.

These are the Python references of src/xplain/web/analyst.js (the browser computation exception, docs/TECHNICAL.md "Run the web page"):
the page computes the same values up to the match clock, and tests/test_analyst_js.py checks it against these,
exactly. Sums run left to right in seq order (analytics.seq_sum), as the page adds, so the floats are equal.

vs expected is the recap card's measure (recap.team_passing): completed passes minus the sum of their xP. Its
running value after a team's last pass of the match equals the card's vs_expected_xp before round2.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .contracts import PassMark

if TYPE_CHECKING:
    from .bundle import BundleMark

# the narration's difficulty tiers (narration.pass_tier), easiest first; Player counts by the same words
TIERS: tuple[str, ...] = ("very easy", "easy", "tricky", "difficult", "very difficult")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingPoint(_Strict):
    """A team's running numbers just after one of its passes."""

    t: float
    seq: int
    passes: int
    completed: int
    expected_xp: float     # the sum of xP so far, unrounded
    vs_expected_xp: float  # completed - expected_xp, unrounded


class TierPassing(_Strict):
    """A team's passes of one difficulty tier: how many, how many completed, the xP they summed to."""

    tier: str
    passes: int
    completed: int
    expected_xp: float
    vs_expected_xp: float


def team_marks(marks: Sequence[PassMark], team_id: str, as_of_t: float | None = None,
               period: int | None = None) -> list[PassMark]:
    """The team's passes with t <= as_of_t (all when None) and in the period (any when None), in seq order."""
    return sorted((m for m in marks if m.team_id == team_id and (as_of_t is None or m.t <= as_of_t)
                   and (period is None or m.period == period)), key=lambda m: m.seq)


def team_swing(marks: Sequence[PassMark], team_id: str, as_of_t: float | None = None,
               period: int | None = None) -> list[SwingPoint]:
    """One point per pass of the team (team_marks): the running completed count, the xP added one pass at a time
    from 0.0, and completed minus that sum."""
    out: list[SwingPoint] = []
    done = 0
    xp = 0.0
    for i, m in enumerate(team_marks(marks, team_id, as_of_t, period), start=1):
        done += int(m.completed)
        xp += m.xp
        out.append(SwingPoint(t=m.t, seq=m.seq, passes=i, completed=done, expected_xp=xp, vs_expected_xp=done - xp))
    return out


def team_tiers(marks: Sequence[BundleMark], team_id: str, as_of_t: float | None = None,
               period: int | None = None) -> list[TierPassing]:
    """The team's passes per tier (mark.tier.en), in TIERS order; a mark without a tier is not counted."""
    by_tier: dict[str, list[BundleMark]] = {t: [] for t in TIERS}
    for m in team_marks(marks, team_id, as_of_t, period):
        if m.tier is not None and m.tier.en in by_tier:
            by_tier[m.tier.en].append(m)
    out: list[TierPassing] = []
    for tier, ps in by_tier.items():
        done = sum(m.completed for m in ps)
        xp = 0.0
        for m in ps:
            xp += m.xp
        out.append(TierPassing(tier=tier, passes=len(ps), completed=done, expected_xp=xp, vs_expected_xp=done - xp))
    return out
