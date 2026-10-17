"""
Narration contract: what the Narrator returns for one Moment, and the difficulty words it is given.

One spoken sentence per language (English, Spanish, Greek) saying the same thing, with no numbers
and no minute (the card shows those, from code), and the ranks of the drivers the lines use. The
Verifier checks it against the MomentBundle before anything is shown.

DIFFICULTY_TIERS turns a Moment's xP into a plain difficulty phrase per language. The cutoffs are
editorial constants chosen from the xP distribution of the outliers over seeds 1-10 (failures sit at
xP 0.920-0.997, median 0.951; successes at 0.112-0.350, median 0.302); change them here only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .analytics import Outlier
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


def difficulty_tier(moment: Moment) -> DifficultyTier:
    """The difficulty tier of a Moment, from its kind and xP."""
    for lowest, tier in DIFFICULTY_TIERS[moment.kind]:
        if moment.xp >= lowest:
            return tier
    raise ValueError(f"no difficulty tier for {moment.kind} at xp={moment.xp}")
