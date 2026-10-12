"""
Narration contract: what the Narrator returns for one Moment.

One sentence per language (English, Spanish, Greek) saying the same thing, every number written
in the lines (exactly as written, digits only) and the ranks of the drivers the lines use. The
Verifier checks it against the MomentBundle before anything is shown.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class NarratorOut(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    line_en: str
    line_es: str
    line_el: str
    numbers_used: list[str]
    drivers_used: list[int]
