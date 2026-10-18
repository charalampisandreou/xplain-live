"""
Pure functions that turn xP and the pass outcome into display values. No I/O.

xP is situational pass difficulty (completion probability from start-of-pass information),
not pass quality. Band and outlier thresholds live in ``AnalyticsConfig`` and are provisional.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from .config import AnalyticsConfig

Band = Literal["routine", "moderate", "hard", "exceptional"]
Outlier = Literal["surprise_success", "surprise_failure"]

DEFAULT = AnalyticsConfig()


def seq_sum(values: Iterable[float]) -> float:
    """Floats added one at a time, left to right, from 0.0: the way the page (player.js) adds. Python's sum()
    corrects for float rounding since 3.12, so it can differ from the page in the last bits; every sum the page
    must match (player.js parity, the recap card's two-decimal values) uses this."""
    total = 0.0
    for v in values:
        total += v
    return total


def percent(x: float) -> int:
    """A probability as a whole percent, the one rounding rule for every displayed percent: round(100 * x)."""
    return round(100 * x)


def difficulty(xp: float) -> int:
    """0 (certain to be completed) to 100 (certain to fail): round(100 * (1 - xP))."""
    return min(max(percent(1.0 - xp), 0), 100)


def difficulty_band(d: int, cfg: AnalyticsConfig = DEFAULT) -> Band:
    if d >= cfg.band_exceptional_from:
        return "exceptional"
    if d >= cfg.band_hard_from:
        return "hard"
    if d >= cfg.band_moderate_from:
        return "moderate"
    return "routine"


def delta(completed: bool, xp: float) -> float:
    """Outcome minus expectation: positive when a pass beat its xP."""
    return int(completed) - xp


def outlier(completed: bool, xp: float, cfg: AnalyticsConfig = DEFAULT) -> Outlier | None:
    if completed and xp <= cfg.surprise_success_max_xp:
        return "surprise_success"
    if not completed and xp >= cfg.surprise_failure_min_xp:
        return "surprise_failure"
    return None
