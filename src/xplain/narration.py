"""
Narration contract: what the Narrator returns for one Moment, and the difficulty words it is given.

One spoken sentence per language (English, Spanish, Greek) saying the same thing, with no numbers
and no minute (the card shows those, from code), and the ranks of the drivers the lines use. The
Verifier checks it against the MomentBundle before anything is shown.

DIFFICULTY_TIERS turns a Moment's xP into a plain difficulty phrase per language. The cutoffs are
editorial constants chosen from the xP distribution of the outliers over seeds 1-10 (failures sit at
xP 0.920-0.997, median 0.951; successes at 0.112-0.350, median 0.302); change them here only.

pass_tier gives ANY pass the same words on one xP scale, for the page (the marks): the failure tiers
from EASY_FROM_XP up, the success tiers below it. EASY_FROM_XP is not a new cutoff: it is the lower
xP of the "hard" difficulty band (AnalyticsConfig.band_hard_from). Every outlier gets the same tier
from pass_tier as from difficulty_tier (failures sit above it, successes below).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .analytics import DEFAULT, Outlier
from .contracts import Moment

Family = Literal["easy", "hard"]


class NarratorOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    line_en: str
    line_es: str
    line_el: str
    drivers_used: list[int]


@dataclass(frozen=True)
class DifficultyTier:
    family: Family   # "easy" for passes that should have come off, "hard" for passes that should not
    en: str
    es: str
    el: str          # agrees with "η πάσα" (feminine)

    def words(self) -> dict[str, str]:
        return {"en": self.en, "es": self.es, "el": self.el}


# Per kind: (lowest xP of the tier, tier), highest tier first; the first tier with xp >= its bound applies.
DIFFICULTY_TIERS: dict[Outlier, tuple[tuple[float, DifficultyTier], ...]] = {
    "surprise_failure": (
        (0.96, DifficultyTier("easy", "very easy", "muy fácil", "πολύ εύκολη")),
        (0.00, DifficultyTier("easy", "easy", "fácil", "εύκολη")),
    ),
    "surprise_success": (
        (0.30, DifficultyTier("hard", "tricky", "complicado", "περίπλοκη")),
        (0.20, DifficultyTier("hard", "difficult", "difícil", "δύσκολη")),
        (0.00, DifficultyTier("hard", "very difficult", "muy difícil", "πολύ δύσκολη")),
    ),
}


EASY_FROM_XP = 1 - DEFAULT.band_hard_from / 100   # 0.60: xP above which a pass is in the easy family


def _tier(kind: Outlier, xp: float) -> DifficultyTier:
    for lowest, tier in DIFFICULTY_TIERS[kind]:
        if xp >= lowest:
            return tier
    raise ValueError(f"no difficulty tier for {kind} at xp={xp}")


def difficulty_tier(moment: Moment) -> DifficultyTier:
    """The difficulty tier of a Moment, from its kind and xP."""
    return _tier(moment.kind, moment.xp)


def pass_tier(xp: float) -> DifficultyTier:
    """The difficulty tier of any pass from its xP alone: the failure tiers at xP >= EASY_FROM_XP, the success
    tiers below (very easy / easy / tricky / difficult / very difficult)."""
    return _tier("surprise_failure" if xp >= EASY_FROM_XP else "surprise_success", xp)
