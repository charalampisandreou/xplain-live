"""
Analyst: turns scored passes into Moments and picks which ones get a card. Deterministic, no I/O.

Difficulty, band, delta and the outlier flag come from analytics.py; this module only packages
them. Highlights are ranked per kind (surprise_success and surprise_failure are never ranked in
one list), by significance = |delta|, ties broken by the earlier seq.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import get_args

from . import analytics
from .analytics import Outlier
from .config import AnalyticsConfig
from .contracts import Moment, PassGeometry
from .schema import PassEvent

OUTLIER_KINDS: tuple[Outlier, ...] = get_args(Outlier)


def match_minute(t: float) -> int:
    """Match minute as shown on a clock: 0-59.99 s is minute 1. Uses t as given (it runs on across halftime)."""
    return math.floor(t / 60.0) + 1


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
) -> Moment | None:
    """Package a scored pass as a Moment, or return None when it is not an outlier."""
    if not 0.0 <= xp <= 1.0:
        raise ValueError(f"xp must be in [0, 1], got {xp!r} for event {event.event_id}")
    completed = event.details.completed
    kind = analytics.outlier(completed, xp, cfg)
    if kind is None:
        return None

    d = analytics.difficulty(xp)
    band = analytics.difficulty_band(d, cfg)
    dlt = analytics.delta(completed, xp)
    minute = match_minute(event.t)
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
        "minute": minute,
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
        minute=minute,
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


def qualifies_live(moment: Moment, earlier: Sequence[Moment], per_kind: int) -> bool:
    """Live gate: True if this moment is in the running top per_kind of its kind, counting the moments so far."""
    top = select_highlights([*earlier, moment], per_kind)[moment.kind]
    return any(m.moment_id == moment.moment_id for m in top)
