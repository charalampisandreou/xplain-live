"""
Analyst: turns scored passes into Moments and ranks them. Deterministic, no I/O.

Difficulty, band, delta and the outlier flag come from analytics.py; this module only packages
them. Highlights are ranked per kind (surprise_success and surprise_failure are never ranked in
one list), by significance = |delta|, ties broken by the earlier seq.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import get_args

from . import analytics
from .analytics import Outlier
from .clock import match_clock
from .config import AnalyticsConfig
from .contracts import Moment, PassGeometry
from .schema import PassEvent

OUTLIER_KINDS: tuple[Outlier, ...] = get_args(Outlier)


def build_evidence(event_id: str, facts: Mapping[str, bool | int | float | str | None]) -> dict[str, bool | int | float | str]:
    """Key every known fact as f"{event_id}:{name}", in the order given; facts that are None are left out."""
    return {f"{event_id}:{name}": value for name, value in facts.items() if value is not None}


def build_moment(
    event: PassEvent,
    xp: float,
    model_version: str,
    *,
    cfg: AnalyticsConfig = analytics.DEFAULT,
    player_names: Mapping[str, str] | None = None,
    team_names: Mapping[str, str] | None = None,
    second_half_start: float | None = None,
) -> Moment | None:
    """Package a scored pass as a Moment, or return None when it is not an outlier.

    second_half_start (t of the second-half kick-off, clock.second_half_start) is needed for period-2 passes.
    """
    if not 0.0 <= xp <= 1.0:
        raise ValueError(f"xp must be in [0, 1], got {xp!r} for event {event.event_id}")
    completed = event.details.completed
    kind = analytics.outlier(completed, xp, cfg)
    if kind is None:
        return None

    d = analytics.difficulty(xp)
    band = analytics.difficulty_band(d, cfg)
    dlt = analytics.delta(completed, xp)
    clock = match_clock(event.t, event.period, second_half_start)
    player_name = (player_names or {}).get(event.player_id) if event.player_id is not None else None
    team_name = (team_names or {}).get(event.team_id)
    geometry = PassGeometry(
        start_x=event.inputs.start_x,
        start_y=event.inputs.start_y,
        end_x=event.details.end_x,
        end_y=event.details.end_y,
        length=event.details.length,
        ball_speed_ms=event.details.ball_speed_ms,
    )
    evidence = build_evidence(event.event_id, {
        "xp": xp,
        "delta": dlt,
        "difficulty": d,
        "difficulty_band": band,
        "completed": completed,
        "minute": clock.minute,
        "minute_added": clock.added,
        "minute_label": clock.label,
        "length": geometry.length,
        "ball_speed_ms": geometry.ball_speed_ms,
        "start_x": geometry.start_x,
        "start_y": geometry.start_y,
        "end_x": geometry.end_x,
        "end_y": geometry.end_y,
        "player_name": player_name,
        "team_name": team_name,
    })
    return Moment(
        moment_id=f"moment:{event.event_id}",
        kind=kind,
        event_id=event.event_id,
        match_id=event.match_id,
        seq=event.seq,
        t=event.t,
        minute=clock.minute,
        minute_added=clock.added,
        minute_label=clock.label,
        period=event.period,
        team_id=event.team_id,
        team_name=team_name,
        player_id=event.player_id,
        player_name=player_name,
        completed=completed,
        xp=xp,
        delta=dlt,
        difficulty=d,
        band=band,
        significance=abs(dlt),
        pass_geometry=geometry,
        evidence=evidence,
        model_version=model_version,
    )


def select_highlights(moments: Sequence[Moment], per_kind: int) -> dict[Outlier, list[Moment]]:
    """Top per_kind moments of each kind, strongest first (ties: earlier seq). Both kinds are always keys."""
    if per_kind < 0:
        raise ValueError(f"per_kind must be >= 0, got {per_kind}")
    return {
        kind: sorted((m for m in moments if m.kind == kind), key=lambda m: (-m.significance, m.seq))[:per_kind]
        for kind in OUTLIER_KINDS
    }


def select_half_highlights(moments: Sequence[Moment], per_kind: int) -> dict[int, dict[Outlier, list[Moment]]]:
    """select_highlights within each half: {1: {kind: top per_kind}, 2: {...}}. Both periods are always keys."""
    return {period: select_highlights([m for m in moments if m.period == period], per_kind) for period in (1, 2)}


def qualifies_live(moment: Moment, earlier: Sequence[Moment], per_kind: int) -> bool:
    """Live gate: True if this moment is in the running top per_kind of its kind, counting the moments so far."""
    top = select_highlights([*earlier, moment], per_kind)[moment.kind]
    return any(m.moment_id == moment.moment_id for m in top)
