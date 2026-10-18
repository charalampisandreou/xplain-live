"""
Verifier: checks a NarratorOut against its MomentBundle. Deterministic, no LLM, no I/O.

Checks (every failing check adds a reason; all reasons are collected, not just the first):
  * sentence:   each line is non-empty, has no line break and holds one sentence
  * length:     each line has at most MAX_WORDS[lang] (= WORD_LIMIT) whitespace-separated words
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
  * outcome_cause: a causal marker (CAUSAL_MARKERS: because, porque, λόγω, ...) may only give a reason
                for the rating, so it must sit next to a difficulty word of the moment's family (at most
                CAUSE_GAP words between them, either side); anywhere else it reads as a cause of the outcome
  * unsupported_fact: every phrase of the phrase table (approved phrases, position pieces and every
                hints suggestion, for any label) found in a line must belong to a driver in drivers_used
  * lead_driver: "mainly" and its kin (LEAD_WORDS) only when driver 1 is narrated and in drivers_used
  * needs_driver: drivers_used is not empty when a narrated driver points the way of the tier
  * driver_expressed: hints mode (strict has the phrases check; free mode shows the model only English labels):
                every driver in drivers_used is named in every line by one of its wordings (phrases.wordings:
                phrase, pieces, hints, recognised forms)
  * role_parity: when drivers_used is not empty, every line gives the drivers a role: a reason for the rating
                (CAUSAL_MARKERS, which outcome_cause keeps next to a difficulty word) or a concession on the
                outcome (CONCESSION_MARKERS); the three lines give the same role
  * one_position_piece: free and hints modes: a line names a position driver by its depth or its width, not both
  * one_role_per_clause: no clause (text between commas, semicolons, colons or dashes) holds both a concession
                marker and a causal marker ("even though it was very easy because it came from deep")
  * tier_first_contrast: a line that opens with its difficulty word ("Easy because ...") has a contrast word
                before the outcome (CONTRAST_WORDS: yet / but, pero, αλλά)
  * rating_tense: no present-tense rating: a present copula (PRESENT_COPULA: is/are, es/son, είναι), an optional
                intensifier, then a difficulty word ("is easy"); the rating is in the past ("was easy")
  * double_contrast: a line holds a contrast word (CONTRAST_WORDS) or a concession marker (CONCESSION_MARKERS),
                not both ("is easy but does not reach a teammate, even though ...")

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
from .phrases import (DEPTH_HINTS, DEPTH_PHRASES, ENGLISH_HEIGHT_HINTS, GREEK_ADJECTIVES, GREEK_DEPTH_HINTS,
                      LABEL_PHRASES, RECOGNISED_DEPTH_FORMS, RECOGNISED_LABEL_FORMS, WIDTH_HINTS, WIDTH_PHRASES,
                      driver_phrases, narrated, wordings)

LANGS: tuple[str, ...] = ("en", "es", "el")

# Word limit: one value for every language (the plate has room for two lines); one place to change it.
# WORD_TARGET is what the prompt asks for; only WORD_LIMIT is checked.
WORD_LIMIT = 24
WORD_TARGET = 18
MAX_WORDS: dict[str, int] = {lang: WORD_LIMIT for lang in LANGS}

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
    "el": (r"δ[υύ]σκολ\w*", r"ζόρικ\w*", r"απαιτητικ\w*", r"περίπλοκ\w*"),
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
        r"\bsuccess\w*",           # the brief forbids "success"; the card shows the outcome
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
        r"\bcon [eé]xito\b", r"\b[eé]xit\w*",
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
        r"επιτυχ\w*",
        r"\bλόγω ότι\b",            # the owner's note: "λόγω του ότι"
        # verbs of intent, effort or credit (present, participle and past forms)
        r"\bεπιχειρ(εί|ούν|ώντας)\b", r"\bεπιχείρησε\b", r"\bπροσπαθ(εί|ούν|ώντας)\b", r"\bπροσπάθησε\b",
        r"\bκαταφέρν(ει|ουν|οντας)\b", r"\bκατάφερε\b",
    ),
}


# Causal markers per language. A marker may give a reason for the model's rating ("difficult because it was
# lofted high"), never for the outcome ("goes astray because ..."); concessions ("even though") are fine.
CAUSAL_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (r"\bbecause\b", r"\bdue to\b", r"\bowing to\b", r"\bthanks to\b", r"\bwhat makes\b",
           r"\bwhich ma(de|kes) it\b"),
    "es": (r"\bporque\b", r"\bdebido a\b", r"\ba causa de\b", r"\bgracias a\b", r"\bpor culpa de\b",
           r"\blo que hace\b", r"\blo que lo hac(e|ía)\b", r"\bya que\b", r"\bpuesto que\b"),
    "el": (r"\bλόγω\b", r"\bεξαιτίας\b", r"\bεπειδή\b", r"\bγιατί\b", r"\bχάρη σ\w*", r"\bαφού\b", r"\bδιότι\b",
           r"\bπου την (έκανε|κάνει)\b"),
}
# Concession markers per language: the driver is conceded against the outcome ("even though it was played along
# the ground"). With CAUSAL_MARKERS they give a driver its role (role_parity).
CONCESSION_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (r"\beven though\b", r"\balthough\b", r"\bthough\b", r"\bdespite\b", r"\bin spite of\b"),
    "es": (r"\baunque\b", r"\ba pesar de\b", r"\bpese a\b"),
    "el": (r"\bπαρ[όο]λο που\b", r"\bπαρ['’ʼ] ?[όο]λο που\b", r"\bαν και\b", r"\bπαρά\b",
           r"\bμολονότι\b", r"\bενώ\b"),
}
CAUSE_GAP = 2   # at most this many words between a causal marker and a difficulty word

# Words that single out the strongest driver: only for driver 1.
LEAD_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\bmainly\b", r"\bmostly\b", r"\babove all\b"),
    "es": (r"\bsobre todo\b", r"\bprincipalmente\b"),
    "el": (r"\bκυρίως\b", r"\bπρωτίστως\b"),
}

# The two vague labels: BANNED rejects their wording already, so the unsupported_fact scan leaves them out.
VAGUE_LABELS = frozenset({"with another part of the body", "from another kind of play"})
WORD = re.compile(r"\w+")


def _compile(table: dict[str, tuple[str, ...]]) -> dict[str, list[re.Pattern[str]]]:
    return {lang: [re.compile(p, re.IGNORECASE) for p in patterns] for lang, patterns in table.items()}


_BANNED_RE = _compile(BANNED)
_NUMBER_WORDS_RE = _compile(NUMBER_WORDS)
_FAMILY_RE = {"easy": _compile(EASY_WORDS), "hard": _compile(HARD_WORDS)}
# A line that opens with its difficulty word (the tier_first frame) needs one of these before the outcome.
CONTRAST_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\byet\b", r"\bbut\b"), "es": (r"\bpero\b",), "el": (r"\bαλλά\b",),
}
# Intensifiers that may come before an opening difficulty word ("Very easy because ...").
OPENING_INTENSIFIERS: dict[str, str] = {
    "en": r"(?:very|especially|extremely|particularly|really|quite)\s+",
    "es": r"(?:muy|especialmente|realmente|bastante)\s+",
    "el": r"(?:πολύ|εξαιρετικά|ιδιαίτερα|αρκετά)\s+",
}
CLAUSE_BREAK = re.compile(r"[,;:\u2013\u2014]")

_CAUSAL_RE = _compile(CAUSAL_MARKERS)
_CONCESSION_RE = _compile(CONCESSION_MARKERS)
_CONTRAST_RE = _compile(CONTRAST_WORDS)
_OPENING_RE = {lang: re.compile(rf"^\W*(?:{p})?", re.IGNORECASE) for lang, p in OPENING_INTENSIFIERS.items()}
_LEAD_RE = _compile(LEAD_WORDS)


def _table_phrases() -> dict[str, list[str]]:
    """Every phrase of the phrase table per language, longest first: label phrases (VAGUE_LABELS left out),
    depth and width pieces, the extra hints wordings (English height forms, Greek adjectives, depth and width
    hints) and the recognised forms (zone nouns, "header", the Greek forms of the owner's examples)."""
    out: dict[str, set[str]] = {lang: set() for lang in LANGS}
    labels = {k: v for k, v in LABEL_PHRASES.items() if k not in VAGUE_LABELS}
    for table in (DEPTH_PHRASES, WIDTH_PHRASES, labels):
        for phrase in table.values():
            for lang in LANGS:
                out[lang].add(phrase[lang])
    out["en"].update(w for words in ENGLISH_HEIGHT_HINTS.values() for w in words)
    out["el"].update(GREEK_ADJECTIVES.values())
    out["el"].update(GREEK_DEPTH_HINTS.values())
    for table in (DEPTH_HINTS, WIDTH_HINTS, RECOGNISED_LABEL_FORMS, RECOGNISED_DEPTH_FORMS):
        for per_lang in table.values():
            for lang, words in per_lang.items():
                out[lang].update(words)
    return {lang: sorted(phrases, key=lambda p: (-len(p), p)) for lang, phrases in out.items()}


TABLE_PHRASES: dict[str, list[str]] = _table_phrases()


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


def _word_index(tokens: list[re.Match[str]], pos: int) -> int:
    """The index of the word at character pos (or the first word after it)."""
    return next((i for i, t in enumerate(tokens) if pos < t.end()), len(tokens))


def check_outcome_cause(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """A causal marker must sit next to a difficulty word of the moment's family: at most CAUSE_GAP words
    between them, on either side ("difficult because", "what makes it difficult"). Elsewhere it reads as a
    cause of the outcome ("goes astray because it was along the ground").

    A heuristic (adjacency to a difficulty word), not a reading of the sentence. It cannot catch a cause of
    the outcome placed right next to a difficulty word ("Weso Wendez's easy pass, lost because it was lofted
    high": two words between "easy" and "because"), nor a cause given without a listed marker ("as", "since").
    """
    family = difficulty_tier(bundle.moment).family
    reasons = []
    for lang, line in _lines(out).items():
        tokens = list(WORD.finditer(line))
        words = [(_word_index(tokens, m.start()), _word_index(tokens, m.end() - 1))
                 for p in _FAMILY_RE[family][lang] for m in p.finditer(line)]
        for pattern in _CAUSAL_RE[lang]:
            for m in pattern.finditer(line):
                first, last = _word_index(tokens, m.start()), _word_index(tokens, m.end() - 1)
                if not any(first - end - 1 <= CAUSE_GAP if first > end else start - last - 1 <= CAUSE_GAP
                           for start, end in words):
                    reasons.append(f"outcome_cause: line_{lang} says {m.group(0)!r} away from a difficulty word: "
                                   f"a reason may explain the rating, never the outcome")
    return reasons


def _phrase_spans(line: str, phrase: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", line, re.IGNORECASE)]


def _driver_wordings(driver: Driver, lang: str) -> list[str]:
    """Every accepted wording of one driver in one language (phrases.wordings): its phrase (and pieces), its hints
    suggestions and its recognised forms."""
    return wordings(driver)["all"][lang]


def check_unsupported_facts(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """Every phrase-table wording in a line (whole words, case-insensitive) must belong to a driver in
    drivers_used: "under pressure" on a pass with no pressure driver, or a narrated driver left out of
    drivers_used, is rejected. Best effort: paraphrases ("with his head", "high in the air") are not caught."""
    drivers = narrated_drivers(bundle)
    used = [drivers[r] for r in dict.fromkeys(out.drivers_used) if r in drivers]
    reasons = []
    for lang, line in _lines(out).items():
        covered = [s for d in used for w in _driver_wordings(d, lang) for s in _phrase_spans(line, w)]
        for phrase in TABLE_PHRASES[lang]:   # longest first, so a piece inside a reported phrase is not repeated
            for a, b in _phrase_spans(line, phrase):
                if not any(x <= a and b <= y for x, y in covered):
                    covered.append((a, b))
                    reasons.append(f"unsupported_fact: line_{lang} says {line[a:b]!r}, "
                                   f"which is not a driver in drivers_used")
    return reasons


def check_lead_driver(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """"mainly" and its kin (LEAD_WORDS) single out the strongest driver: only when driver 1 is narrated and used."""
    if 1 in narrated_drivers(bundle) and 1 in out.drivers_used:
        return []
    return [f"lead_driver: line_{lang} says {m.group(0)!r}, but driver 1 is not a narrated driver in drivers_used"
            for lang, line in _lines(out).items() for p in _LEAD_RE[lang] if (m := p.search(line))]


def check_needs_driver(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """When a narrated driver points the way of the tier, the line must use at least one driver."""
    tier = difficulty_tier(bundle.moment)
    wanted = "raises" if tier.family == "easy" else "lowers"
    matching = [rank for rank, d in narrated_drivers(bundle).items() if d.direction == wanted]
    if out.drivers_used or not matching:
        return []
    return [f"needs_driver: drivers_used is empty, but driver(s) {matching} point the way of the rating "
            f"({wanted}) and the pass is rated {tier.family} ({tier.en})"]


def _used_drivers(out: NarratorOut, bundle: MomentBundle) -> list[Driver]:
    drivers = narrated_drivers(bundle)
    return [drivers[r] for r in dict.fromkeys(out.drivers_used) if r in drivers]


def check_driver_expressed(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """Every driver in drivers_used is named in every line by one of its wordings (phrases.wordings). Hints mode,
    whose facts list those wordings; strict mode has the stricter phrases check, and free mode shows the model only
    the English labels. A paraphrase that is not listed is rejected and retried."""
    reasons = []
    for d in _used_drivers(out, bundle):
        for lang, line in _lines(out).items():
            if not any(has_phrase(line, w) for w in _driver_wordings(d, lang)):
                reasons.append(f"driver_expressed: line_{lang} lists driver {d.rank} ({d.label}) but does not name "
                               f"it; every line names every driver in drivers_used")
    return reasons


def line_roles(lang: str, line: str) -> set[str]:
    """The roles a line gives its drivers: "reason" (a causal marker) and/or "concession" (a concession marker)."""
    roles = set()
    if any(p.search(line) for p in _CAUSAL_RE[lang]):
        roles.add("reason")
    if any(p.search(line) for p in _CONCESSION_RE[lang]):
        roles.add("concession")
    return roles


def check_role_parity(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """When a line uses drivers, each line gives them a role (the reason for the rating, or a concession on the
    outcome), and the three lines give the same one: a driver is never a bare description in one language
    ("Η ψηλή πάσα ... βρίσκει συμπαίκτη") while another language gives the reason."""
    if not _used_drivers(out, bundle):
        return []
    roles = {lang: line_roles(lang, line) for lang, line in _lines(out).items()}
    reasons = [f"role_parity: line_{lang} names a driver but gives it no role: say it as the reason for the rating "
               f"(next to the difficulty word) or as a concession (even though)"
               for lang, r in roles.items() if not r]
    given = {lang: r for lang, r in roles.items() if r}
    if len({frozenset(r) for r in given.values()}) > 1:
        said = ", ".join(f"line_{lang} {'+'.join(sorted(r))}" for lang, r in given.items())
        reasons.append(f"role_parity: the lines give the drivers different roles ({said}); all three take the "
                       f"same one")
    return reasons


def check_one_position_piece(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """A position driver is named by its depth or by its width, not both, in each line (free and hints modes)."""
    reasons = []
    for d in _used_drivers(out, bundle):
        if d.feature != "position":
            continue
        parts = wordings(d)
        for lang, line in _lines(out).items():
            if all(any(has_phrase(line, w) for w in parts[piece][lang]) for piece in ("depth", "width")):
                reasons.append(f"one_position_piece: line_{lang} names both the depth and the width of driver "
                               f"{d.rank}; use one of them")
    return reasons


def check_one_role_per_clause(out: NarratorOut) -> list[str]:
    """No clause gives a concession and a reason at once: "even though it was very easy because it came from deep"
    is rejected; "Even though it was lofted high, the pass was difficult because ..." keeps them in two clauses."""
    reasons = []
    for lang, line in _lines(out).items():
        for clause in CLAUSE_BREAK.split(line):
            if any(p.search(clause) for p in _CONCESSION_RE[lang]) and any(p.search(clause) for p in _CAUSAL_RE[lang]):
                reasons.append(f"one_role_per_clause: line_{lang} gives a concession and a reason in one clause "
                               f"({clause.strip()!r}); one role per clause")
    return reasons


def check_tier_first_contrast(out: NarratorOut) -> list[str]:
    """A line that opens with a difficulty word ("Easy because it was played along the ground, yet ...") has a
    contrast word (CONTRAST_WORDS) before the outcome."""
    reasons = []
    for lang, line in _lines(out).items():
        rest = line[_OPENING_RE[lang].match(line).end():]
        opens_with_tier = any(p.match(rest) for table in _FAMILY_RE.values() for p in table[lang])
        if opens_with_tier and not any(p.search(line) for p in _CONTRAST_RE[lang]):
            words = " / ".join(w.replace("\\b", "") for w in CONTRAST_WORDS[lang])
            reasons.append(f"tier_first_contrast: line_{lang} opens with the difficulty word but has no contrast "
                           f"word ({words}) before the outcome")
    return reasons


# A present-tense copula before a difficulty word rates the pass in the present ("is easy"); the rating is in the
# past ("was easy", "era fácil", "ήταν εύκολη"). English "'s" is left out: it is mostly the possessive.
PRESENT_COPULA: dict[str, str] = {
    "en": r"\b(?:is|are)\s+", "es": r"\b(?:es|son)\s+", "el": r"\bείναι\s+",
}
_PRESENT_COPULA_RE = {lang: re.compile(p, re.IGNORECASE) for lang, p in PRESENT_COPULA.items()}
_INTENSIFIER_RE = {lang: re.compile(p, re.IGNORECASE) for lang, p in OPENING_INTENSIFIERS.items()}


def check_rating_tense(out: NarratorOut) -> list[str]:
    """No present-tense rating: a present copula (PRESENT_COPULA), an optional intensifier, then a difficulty word
    of either family ("is easy", "es muy fácil", "είναι δύσκολη")."""
    reasons = []
    for lang, line in _lines(out).items():
        for m in _PRESENT_COPULA_RE[lang].finditer(line):
            rest = line[m.end():]
            if (i := _INTENSIFIER_RE[lang].match(rest)) is not None:
                rest = rest[i.end():]
            if any(p.match(rest) for table in _FAMILY_RE.values() for p in table[lang]):
                end = len(line) - len(rest) + len(WORD.match(rest).group(0))   # through the difficulty word
                said = line[m.start():end]
                reasons.append(f"rating_tense: line_{lang} rates the pass in the present ({said.strip()!r}); "
                               f"the rating is in the past")
                break
    return reasons


def check_double_contrast(out: NarratorOut) -> list[str]:
    """One contrast per line: not both a contrast word (CONTRAST_WORDS: but / yet, pero, αλλά) and a concession
    marker (CONCESSION_MARKERS: even though, aunque, παρόλο που ...)."""
    reasons = []
    for lang, line in _lines(out).items():
        contrast = next((m.group(0) for p in _CONTRAST_RE[lang] if (m := p.search(line))), None)
        concession = next((m.group(0) for p in _CONCESSION_RE[lang] if (m := p.search(line))), None)
        if contrast and concession:
            reasons.append(f"double_contrast: line_{lang} has both {contrast!r} and {concession!r}; one contrast "
                           f"per line")
    return reasons


def verify(out: NarratorOut, bundle: MomentBundle, mode: str | None = None) -> VerifierResult:
    """Run every check and collect every failing reason; ok when there are none. Notes never fail a line.

    mode is the phrase mode (default config.phrase_mode()); the phrases check runs in "strict" mode only,
    driver_expressed in "hints" mode, one_position_piece in "free" and "hints" modes.
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
        *check_outcome_cause(out, bundle),
        *check_unsupported_facts(out, bundle),
        *check_lead_driver(out, bundle),
        *check_needs_driver(out, bundle),
        *(check_driver_expressed(out, bundle) if mode == "hints" else []),
        *check_role_parity(out, bundle),
        *(check_one_position_piece(out, bundle) if mode != "strict" else []),
        *check_one_role_per_clause(out),
        *check_tier_first_contrast(out),
        *check_rating_tense(out),
        *check_double_contrast(out),
    ]
    return VerifierResult(ok=not reasons, reasons=reasons, notes=notes)
