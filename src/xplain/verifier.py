"""
Verifier: checks a NarratorOut against its MomentBundle. Deterministic, no LLM, no I/O.

Checks (every failing check adds a reason; all reasons are collected, not just the first):
  * sentence: each line is non-empty, has no line break and holds one sentence
  * length:   each line has at most MAX_WORDS[lang] whitespace-separated words
  * numbers:  every digit run in a line (\\d+, so Greek "3ο" reads as 3) is listed in numbers_used,
              and every numbers_used entry is one of allowed_numbers(bundle)
  * banned:   no line matches a BANNED pattern of its language (claims we have no data for,
              model internals, wrong vocabulary)
  * drivers:  drivers_used has no duplicates and only ranks of the bundle's drivers (empty when
              there are none)

Reasons start with the check name, e.g. "numbers: line_en has 7, not in numbers_used".
"""
from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from .analytics import percent
from .contracts import MomentBundle
from .narration import NarratorOut

LANGS: tuple[str, ...] = ("en", "es", "el")

# Word limit per language; one place to change it.
MAX_WORDS: dict[str, int] = {"en": 18, "es": 22, "el": 22}

DIGITS = re.compile(r"\d+")
# A sentence end followed by more text means a second sentence. Greek uses ";" (and U+037E) as its question mark.
SENTENCE_BREAK: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"[.!?](?=\s+\S)"),
    "es": re.compile(r"[.!?](?=\s+\S)"),
    "el": re.compile(r"[.!?;;](?=\s+\S)"),
}

# Banned claims per language, matched case-insensitively. Keep each list short; add a pattern to ban a phrase.
# Families: box/penalty area, line-breaking, defender counts, left/right, model internals, wrong vocabulary
# ("throw"; none for Spanish, where "lanza un pase" is normal commentary).
BANNED: dict[str, tuple[str, ...]] = {
    "en": (
        r"\binside the (penalty )?box\b", r"\bin the box\b", r"\bpenalty area\b",
        r"\bline[- ]?break\w*", r"\bbreaks? (the|their) lines?\b",
        r"\b(\d+|two|three|four|five|six|several) (defenders|opponents)\b",
        r"\bleft\b", r"\bright\b",
        r"\bshap\b", r"\bbaseline\b", r"\braw score\b", r"\bexpected pass value\b",
        r"\bthrow(?!-ins?\b)\w*",  # "throw-in" is a play pattern, not a throw
    ),
    "es": (
        r"\bdentro del [aá]rea\b", r"\ben el [aá]rea\b",
        r"\brompe(n)? (la|las) l[ií]neas?\b", r"\brompel[ií]neas\b",
        r"\b(\d+|dos|tres|cuatro|cinco|seis|varios) (defensas|defensores|rivales)\b",
        r"\bizquierd[ao]s?\b", r"\bderech[ao]s?\b",
        r"\bshap\b", r"\bl[ií]nea base\b", r"\bpuntuaci[oó]n bruta\b", r"\bvalor esperado del? pase\b",
    ),
    "el": (
        r"μέσα στην περιοχή", r"στη μεγάλη περιοχή",
        r"σπάει τ(η|ις) γραμμ(ή|ές)",
        r"(\d+|δύο|τρεις|τέσσερις|πέντε|έξι|πολλούς) (αμυντικούς|αντιπάλους)",
        r"αριστερ\w*", r"δεξι\w*",
        r"\bshap\b", r"γραμμή βάσης", r"ακατέργαστ\w* βαθμολογ\w*", r"αναμενόμεν\w* αξία",
        r"ρίχν\w*", r"πετάει",
    ),
}
_BANNED_RE: dict[str, list[re.Pattern[str]]] = {
    lang: [re.compile(p, re.IGNORECASE) for p in patterns] for lang, patterns in BANNED.items()
}


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    reasons: list[str]


def allowed_numbers(bundle: MomentBundle) -> frozenset[str]:
    """The numbers a line may contain, as written: the xP percent, the match minute and the difficulty, from the evidence."""
    m = bundle.moment
    ev = m.evidence
    allowed: set[str] = set()
    if (xp := ev.get(f"{m.event_id}:xp")) is not None:
        allowed.add(str(percent(xp)))
    if (minute := ev.get(f"{m.event_id}:minute")) is not None:
        allowed.add(str(minute))
    if (difficulty := ev.get(f"{m.event_id}:difficulty")) is not None:
        allowed.add(str(difficulty))
    return frozenset(allowed)


def _lines(out: NarratorOut) -> dict[str, str]:
    return {lang: getattr(out, f"line_{lang}") for lang in LANGS}


def check_sentence(out: NarratorOut) -> list[str]:
    reasons = []
    for lang, line in _lines(out).items():
        if not line.strip():
            reasons.append(f"sentence: line_{lang} is empty")
        elif "\n" in line or "\r" in line:
            reasons.append(f"sentence: line_{lang} has a line break")
        elif SENTENCE_BREAK[lang].search(line.strip()):
            reasons.append(f"sentence: line_{lang} has more than one sentence")
    return reasons


def check_length(out: NarratorOut) -> list[str]:
    reasons = []
    for lang, line in _lines(out).items():
        n = len(line.split())
        if n > MAX_WORDS[lang]:
            reasons.append(f"length: line_{lang} has {n} words, max {MAX_WORDS[lang]}")
    return reasons


def check_numbers(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    reasons = []
    used = set(out.numbers_used)
    for lang, line in _lines(out).items():
        for n in dict.fromkeys(DIGITS.findall(line)):
            if n not in used:
                reasons.append(f"numbers: line_{lang} has {n}, not in numbers_used")
    allowed = allowed_numbers(bundle)
    for n in dict.fromkeys(out.numbers_used):
        if n not in allowed:
            reasons.append(f"numbers: numbers_used has {n!r}, not in the evidence (allowed {sorted(allowed, key=int)})")
    return reasons


def check_banned(out: NarratorOut) -> list[str]:
    reasons = []
    for lang, line in _lines(out).items():
        for pattern in _BANNED_RE[lang]:
            if match := pattern.search(line):
                reasons.append(f"banned: line_{lang} says {match.group(0)!r}")
    return reasons


def check_drivers(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    reasons = []
    ranks = {d.rank for d in bundle.explanation.drivers}
    dupes = sorted(r for r, c in Counter(out.drivers_used).items() if c > 1)
    if dupes:
        reasons.append(f"drivers: drivers_used repeats {dupes}")
    if not ranks and out.drivers_used:
        reasons.append(f"drivers: the moment has no drivers but drivers_used is {out.drivers_used}")
    elif unknown := sorted(set(out.drivers_used) - ranks):
        reasons.append(f"drivers: drivers_used has {unknown}, the moment's drivers are {sorted(ranks)}")
    return reasons


def verify(out: NarratorOut, bundle: MomentBundle) -> VerifierResult:
    """Run every check and collect every failing reason; ok when there are none."""
    reasons = [
        *check_sentence(out),
        *check_length(out),
        *check_numbers(out, bundle),
        *check_banned(out),
        *check_drivers(out, bundle),
    ]
    return VerifierResult(ok=not reasons, reasons=reasons)
