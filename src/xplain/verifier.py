"""
Verifier: checks a NarratorOut against its MomentBundle. Deterministic, no LLM, no I/O.

Checks (every failing check adds a reason; all reasons are collected, not just the first):
  * sentence:   each line is non-empty, has no line break and holds one sentence
  * length:     each line has at most MAX_WORDS[lang] whitespace-separated words
  * no_numbers: no line has a digit, a "%" or a NUMBER_WORDS word of its language (percent, minute);
                the card shows the numbers and the minute, from code
  * difficulty: no line contradicts the moment's difficulty tier (narration.difficulty_tier): an
                easy-tier moment never gets a HARD_WORDS word, a hard-tier moment never an EASY_WORDS word
  * banned:     no line matches a BANNED pattern of its language (claims we have no data for,
                model internals, wrong vocabulary, the two vague driver labels); a match inside an
                earlier match is not reported twice
  * drivers:    drivers_used has no duplicates and only ranks of the narrated drivers (phrases.narrated:
                the drivers the model is shown; empty when there are none)
  * phrases:    strict phrase mode only (config.phrase_mode): every driver in drivers_used has its
                approved phrase (phrases.driver_phrases) for that language in the line,
                case-insensitive; a position driver may use the composed phrase or one of its pieces
  * driver_direction: every driver in drivers_used, whatever its rank, points the same way as the
                difficulty tier: "raises" for an easy tier, "lowers" for a hard one

Reasons start with the check name, e.g. "no_numbers: line_en has '94%'". Notes are observations that
do not fail a line, e.g. "difficulty: line_en has no difficulty word".
"""
from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from .config import phrase_mode
from .contracts import Driver, MomentBundle
from .narration import NarratorOut, difficulty_tier
from .phrases import driver_phrases, narrated

LANGS: tuple[str, ...] = ("en", "es", "el")

# Word limit per language; one place to change it.
MAX_WORDS: dict[str, int] = {"en": 18, "es": 22, "el": 22}

# A sentence end followed by more text means a second sentence. Greek uses ";" (and U+037E) as its question mark.
SENTENCE_BREAK: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"[.!?](?=\s+\S)"),
    "es": re.compile(r"[.!?](?=\s+\S)"),
    "el": re.compile(r"[.!?;;](?=\s+\S)"),
}

NUMBER_CHARS = re.compile(r"[\d%]+")
# Number and time words per language, matched case-insensitively as stems. Greek "λεπτ" is limited to the
# noun forms (λεπτό, λεπτά, λεπτού, λεπτών), so "λεπτή πάσα" (a delicate pass) stays allowed.
NUMBER_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\bper ?cent\w*", r"\bminutes?\b"),
    "es": (r"\bpor ?ciento\b", r"\bporcentaje\w*", r"\bminut\w*"),
    "el": (r"τοις εκατό", r"ποσοστ\w*", r"λεπτ(ό|ού|ά|ών)\b"),
}

# Difficulty words per language, as stems. A line about an easy-tier moment must not use a HARD word, and the
# other way round. Keep the lists short; every tier word in narration.DIFFICULTY_TIERS must match its family.
EASY_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\beas(y|ier|iest|ily)\b", r"\bsimple\b", r"\broutine\b", r"\bstraightforward\b"),
    "es": (r"\bf[aá]cil\w*", r"\bsencill\w*", r"\bsimple\b", r"\brutinari\w*"),
    "el": (r"ε[υύ]κολ\w*", r"\bαπλ(ή|ής|ό|ός|ές)\b", r"ρουτίν\w*"),
}
HARD_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\btough\w*", r"\bdifficult\w*", r"\bhard(er|est)?\b", r"\btricky\b", r"\bdemanding\b"),
    "es": (r"\bdif[ií]cil\w*", r"\bcomplicad\w*", r"\bcomplej\w*", r"\bexigente\w*"),
    "el": (r"δ[υύ]σκολ\w*", r"ζόρικ\w*", r"απαιτητικ\w*"),
}

# Banned claims per language, matched case-insensitively. Keep each list short; add a pattern to ban a phrase.
# Families: box/penalty area, line-breaking, defender counts, left/right, model internals, wrong vocabulary
# ("throw"; none for Spanish, where "lanza un pase" is normal commentary), and the wording of the two vague
# driver labels (phrases.NOT_NARRATED), which must never reach a line. Within a family the longer patterns
# come first, so a reason quotes the longest match.
BANNED: dict[str, tuple[str, ...]] = {
    "en": (
        r"\binside the (penalty )?box\b", r"\bin the box\b", r"\bpenalty (area|box)\b",
        r"\bthe box\b", r"\bthe area\b",
        r"\bline[- ]?break\w*", r"\bbreaks? (the|their) lines?\b",
        r"\b(\d+|two|three|four|five|six|several) (defenders|opponents)\b",
        r"\bleft\b", r"\bright\b",
        r"\bshap\b", r"\bbaseline\b", r"\braw score\b", r"\bexpected pass value\b",
        r"\bthrow(?!-ins?\b)\w*",  # "throw-in" is a play pattern, not a throw
        r"\banother part of the body\b", r"\banother kind of play\b",
        # verbs of intent, effort or credit (the model rates difficulty; it does not judge the player)
        r"\btr(y|ies|ied|ying)\b", r"\battempt(s|ed|ing)?\b", r"\bpull(s|ed|ing)? off\b",
        r"\bthread(s|ed|ing)?\b", r"\bmanag(e|es|ed|ing) to\b",
    ),
    "es": (
        r"\bdentro del [aá]rea\b", r"\ben el [aá]rea\b", r"\bcerca del [aá]rea\b", r"\bdel [aá]rea\b",
        r"\b[aá]reas?\b",
        r"\brompe(n)? (la|las) l[ií]neas?\b", r"\brompel[ií]neas\b",
        r"\b(\d+|dos|tres|cuatro|cinco|seis|varios) (defensas|defensores|rivales)\b",
        r"\bizquierd[ao]s?\b", r"\bderech[ao]s?\b",
        r"\bshap\b", r"\bl[ií]nea base\b", r"\bpuntuaci[oó]n bruta\b", r"\bvalor esperado del? pase\b",
        r"\botra parte del cuerpo\b", r"\botro tipo de jugada\b",
        # verbs of intent, effort or credit; "se trata de" (it is a ...) is not one
        r"\bintent(a|an|ó|aba|ando|ar)\b", r"(?<!\bse )\btrat(a|an|ó|aba|ando|ar) de\b",
        r"\blogr(a|an|ó|aba|ando|ar)\b",
    ),
    "el": (
        r"μέσα στην περιοχή", r"στη μεγάλη περιοχή", r"\bπεριοχ\w*",
        r"σπάει τ(η|ις) γραμμ(ή|ές)",
        r"(\d+|δύο|τρεις|τέσσερις|πέντε|έξι|πολλούς) (αμυντικούς|αντιπάλους)",
        r"αριστερ\w*", r"δεξι\w*",
        r"\bshap\b", r"γραμμή βάσης", r"ακατέργαστ\w* βαθμολογ\w*", r"αναμενόμεν\w* αξία",
        r"ρίχν\w*", r"πετάει",
        r"\bάλλο μέρος του σώματος\b", r"\bάλλο είδος φάσης\b",
        # verbs of intent, effort or credit (present, participle and past forms)
        r"\bεπιχειρ(εί|ούν|ώντας)\b", r"\bεπιχείρησε\b", r"\bπροσπαθ(εί|ούν|ώντας)\b", r"\bπροσπάθησε\b",
        r"\bκαταφέρν(ει|ουν|οντας)\b", r"\bκατάφερε\b",
    ),
}


def _compile(table: dict[str, tuple[str, ...]]) -> dict[str, list[re.Pattern[str]]]:
    return {lang: [re.compile(p, re.IGNORECASE) for p in patterns] for lang, patterns in table.items()}


_BANNED_RE = _compile(BANNED)
_NUMBER_WORDS_RE = _compile(NUMBER_WORDS)
_FAMILY_RE = {"easy": _compile(EASY_WORDS), "hard": _compile(HARD_WORDS)}


class VerifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    reasons: list[str]
    notes: list[str] = []   # observations that do not fail the line


def difficulty_words(lang: str, line: str) -> dict[str, list[str]]:
    """The easy and hard words a line uses, per family."""
    return {family: [m.group(0) for p in table[lang] if (m := p.search(line))]
            for family, table in _FAMILY_RE.items()}


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


def check_no_numbers(out: NarratorOut) -> list[str]:
    reasons = []
    for lang, line in _lines(out).items():
        for chars in dict.fromkeys(NUMBER_CHARS.findall(line)):
            reasons.append(f"no_numbers: line_{lang} has {chars!r}")
        for pattern in _NUMBER_WORDS_RE[lang]:
            if match := pattern.search(line):
                reasons.append(f"no_numbers: line_{lang} says {match.group(0)!r}")
    return reasons


def check_difficulty(out: NarratorOut, bundle: MomentBundle) -> tuple[list[str], list[str]]:
    """(reasons, notes): a word of the opposite family fails the line; a line with no difficulty word is noted."""
    tier = difficulty_tier(bundle.moment)
    opposite = "hard" if tier.family == "easy" else "easy"
    reasons, notes = [], []
    for lang, line in _lines(out).items():
        found = difficulty_words(lang, line)
        for word in found[opposite]:
            reasons.append(f"difficulty: line_{lang} says {word!r}, but the pass is rated {tier.family} ({tier.en})")
        if not found["easy"] and not found["hard"]:
            notes.append(f"difficulty: line_{lang} has no difficulty word")
    return reasons, notes


def check_banned(out: NarratorOut) -> list[str]:
    reasons = []
    for lang, line in _lines(out).items():
        spans: list[tuple[int, int]] = []
        for pattern in _BANNED_RE[lang]:
            match = pattern.search(line)
            if match and not any(a <= match.start() and match.end() <= b for a, b in spans):
                spans.append(match.span())
                reasons.append(f"banned: line_{lang} says {match.group(0)!r}")
    return reasons


def narrated_drivers(bundle: MomentBundle) -> dict[int, Driver]:
    """The drivers the model is shown, by their original rank (phrases.narrated: NOT_NARRATED and the
    lower-ranked driver of an INCOMPATIBLE_PAIRS pair left out)."""
    return {d.rank: d for d in narrated(bundle.explanation.drivers)}


def check_drivers(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    reasons = []
    ranks = set(narrated_drivers(bundle))
    dupes = sorted(r for r, c in Counter(out.drivers_used).items() if c > 1)
    if dupes:
        reasons.append(f"drivers: drivers_used repeats {dupes}")
    if not ranks and out.drivers_used:
        reasons.append(f"drivers: the moment has no narrated drivers but drivers_used is {out.drivers_used}")
    elif unknown := sorted(set(out.drivers_used) - ranks):
        reasons.append(f"drivers: drivers_used has {unknown}, the narrated drivers are {sorted(ranks)}")
    return reasons


def has_phrase(line: str, phrase: str) -> bool:
    """The phrase appears in the line as whole words, case-insensitive ("raso" is not found in "fracaso")."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", line, re.IGNORECASE) is not None


def check_phrases(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """Each driver in drivers_used has its approved phrase in every line (position: composed phrase or a piece)."""
    reasons = []
    drivers = narrated_drivers(bundle)
    for rank in dict.fromkeys(out.drivers_used):
        if rank not in drivers:
            continue   # check_drivers reports it
        phrases = driver_phrases(drivers[rank])   # the phrase, plus depth and width for a position driver
        for lang, line in _lines(out).items():
            options = [p[lang] for p in phrases.values()]
            if not any(has_phrase(line, o) for o in options):
                wanted = " or ".join(repr(o) for o in options)
                reasons.append(f"phrases: line_{lang} lists driver {rank} but does not say {wanted}")
    return reasons


def check_driver_direction(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """Every used driver points the way of the tier: "raises" for an easy pass, "lowers" for a hard one."""
    tier = difficulty_tier(bundle.moment)
    wanted = "raises" if tier.family == "easy" else "lowers"
    drivers = narrated_drivers(bundle)
    return [
        f"driver_direction: driver {rank} ({drivers[rank].label}) {drivers[rank].direction} the rating, but the "
        f"pass is rated {tier.family} ({tier.en})"
        for rank in dict.fromkeys(out.drivers_used)
        if rank in drivers and drivers[rank].direction != wanted
    ]


def verify(out: NarratorOut, bundle: MomentBundle, mode: str | None = None) -> VerifierResult:
    """Run every check and collect every failing reason; ok when there are none. Notes never fail a line.

    mode is the phrase mode (default config.phrase_mode()); the phrases check runs in "strict" mode only.
    """
    mode = mode or phrase_mode()
    difficulty_reasons, notes = check_difficulty(out, bundle)
    reasons = [
        *check_sentence(out),
        *check_length(out),
        *check_no_numbers(out),
        *difficulty_reasons,
        *check_banned(out),
        *check_drivers(out, bundle),
        *(check_phrases(out, bundle) if mode == "strict" else []),
        *check_driver_direction(out, bundle),
    ]
    return VerifierResult(ok=not reasons, reasons=reasons, notes=notes)
