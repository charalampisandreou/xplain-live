"""
Match clock: the minute a fan sees for an event, from its t, its period and the second-half start.

t runs on from kick-off without a half-time gap, so the second half is timed from its own kick-off.
Seconds 0-59 of a half are its first minute. Time past the half length is added time:
45:00 to 45:59 of the first half is "45+1", and the same past 90 in the second half ("90+N").
The half length is the generator's (GenConfig.half_length_s), read here, not assumed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .config import GenConfig

HALF_LENGTH_S: float = GenConfig().half_length_s
MINUTE_MS = 60_000


def _ms(seconds: float) -> int:
    """Seconds as whole milliseconds (the clock's resolution; the generator writes t to 0.01 s)."""
    return round(seconds * 1000)


class MatchClock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minute: Annotated[int, Field(ge=1)]   # base minute, capped at the end of the half (45 or 90)
    added: Annotated[int, Field(ge=0)]    # added-time minute, 0 in regular time
    label: str                            # "23", "45+2", "90+4"


class Timed(Protocol):
    t: float
    period: int


def clock_label(minute: int, added: int) -> str:
    return f"{minute}+{added}" if added else str(minute)


def match_clock(t: float, period: int, second_half_start: float | None,
                half_length_s: float = HALF_LENGTH_S) -> MatchClock:
    """The clock for an event at t in period 1 or 2. second_half_start is needed for period 2 only.

    Times are counted in whole milliseconds, so the arithmetic is exact: in floats,
    5544.15 - 2844.15 is 2699.999..., which would put a pass at 90:00 in minute 90 instead of 90+1.
    """
    half_ms = _ms(half_length_s)
    half_minutes = half_ms // MINUTE_MS
    if period == 1:
        offset_ms, before = _ms(t), 0
    elif period == 2:
        if second_half_start is None or _ms(t) < _ms(second_half_start):
            raise ValueError(f"period-2 event at t={t} needs a second_half_start <= t, got {second_half_start}")
        offset_ms, before = _ms(t) - _ms(second_half_start), half_minutes
    else:
        raise ValueError(f"period must be 1 or 2, got {period!r}")
    if offset_ms < half_ms:
        minute, added = before + offset_ms // MINUTE_MS + 1, 0
    else:
        minute, added = before + half_minutes, (offset_ms - half_ms) // MINUTE_MS + 1
    return MatchClock(minute=minute, added=added, label=clock_label(minute, added))


def second_half_start(events: Iterable[Timed]) -> float | None:
    """t of the first period-2 event (the second-half kick-off), or None when the match has no second half yet."""
    return min((e.t for e in events if e.period == 2), default=None)


def as_of_t(events: Iterable[Timed], minute: int) -> float:
    """as_of_t for "up to and including match minute N": the t of the last event whose base minute is <= N.

    Base minute, so N = 45 includes the first half's added time and N = 90 the whole match.
    -inf when no event is that early.
    """
    events = list(events)
    start = second_half_start(events)
    return max((e.t for e in events if match_clock(e.t, e.period, start).minute <= minute), default=-math.inf)
