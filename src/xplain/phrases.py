"""
Approved narration phrases for the Explainer's driver labels, in English, Spanish and Greek.

The Narrator gets these phrases instead of the labels and must use them verbatim; the Verifier checks
that every driver a line claims to use has its phrase in that line. The labels themselves (what the
card shows) are unchanged.

Keys:
  * LABEL_PHRASES: the exact label strings the Explainer emits for height, body part, play pattern
    and pressure (explainer.HEIGHT_LABELS, BODY_PART_LABELS, PLAY_PATTERN_LABELS, PRESSURE_LABELS).
  * DEPTH_PHRASES / WIDTH_PHRASES: the zone ids of explainer.depth_zone and explainer.width_zone. A
    position driver's value is "<depth>|<width>"; its phrase is "<depth phrase>, <width phrase>".

NOT_NARRATED holds the labels the model never sees: default conditions and two vague labels.
INCOMPATIBLE_PAIRS holds label pairs that read as a contradiction in one line; when both are narrated,
the lower-ranked one (higher rank number) is left out. narrated() applies both filters, so the
Narrator's facts and the Verifier agree on which drivers the model was shown.

In "hints" phrase mode the model gets suggestions (suggestions()): per language a list of approved
wordings it may use, adapt or replace; for a position driver the two pieces (depth, width) separately,
and for the three height labels an English adjective form (ENGLISH_HEIGHT_HINTS) and the Greek
adjective form (GREEK_ADJECTIVES) as extra options.
"""
from __future__ import annotations

from .contracts import Driver

Phrase = dict[str, str]   # {"en": ..., "es": ..., "el": ...}

DEPTH_PHRASES: dict[str, Phrase] = {
    "defensive_third": {"en": "from their own defensive third", "es": "desde su propio tercio defensivo",
                        "el": "στην άμυνα"},
    "middle_third": {"en": "in the middle third", "es": "en el tercio medio", "el": "στο κέντρο"},
    "attacking_third": {"en": "in the attacking third", "es": "desde el tercio de ataque", "el": "στο επιθετικό τρίτο"},
    "near_goal": {"en": "close to the opposition goal", "es": "cerca de la portería rival",
                  "el": "κοντά στην αντίπαλη εστία"},
}
WIDTH_PHRASES: dict[str, Phrase] = {
    "central": {"en": "down the middle", "es": "por el centro", "el": "από τον άξονα"},
    "off_centre": {"en": "slightly off-centre", "es": "ligeramente descentrado", "el": "ελαφρώς εκτός άξονα"},
    "out_wide": {"en": "out wide", "es": "desde la banda", "el": "από τα πλάγια"},
}
LABEL_PHRASES: dict[str, Phrase] = {
    # height
    "along the ground": {"en": "along the ground", "es": "raso", "el": "συρτά"},
    "played low": {"en": "played low", "es": "a media altura", "el": "χαμηλά"},
    "lofted high": {"en": "lofted high", "es": "por alto", "el": "ψηλά"},
    # body part
    "with the foot": {"en": "with the foot", "es": "con el pie", "el": "με το πόδι"},
    "with a header": {"en": "with a header", "es": "de cabeza", "el": "με κεφαλιά"},
    "from the goalkeeper's hands": {"en": "from the goalkeeper's hands", "es": "desde las manos del portero",
                                    "el": "με τα χέρια του τερματοφύλακα"},
    "with another part of the body": {"en": "with another part of the body", "es": "con otra parte del cuerpo",
                                      "el": "με άλλο μέρος του σώματος"},
    # play pattern
    "in open play": {"en": "in open play", "es": "en juego corrido", "el": "σε ανοιχτό παιχνίδι"},
    "from the kick-off": {"en": "from the kick-off", "es": "de saque de centro", "el": "από τη σέντρα"},
    "from a throw-in": {"en": "from a throw-in", "es": "de saque de banda", "el": "από πλάγιο άουτ"},
    "from a free kick": {"en": "from a free kick", "es": "de falta", "el": "από φάουλ"},
    "from a goal kick": {"en": "from a goal kick", "es": "de saque de puerta", "el": "από ελεύθερο"},
    "in a move started by the goalkeeper": {"en": "in a move started by the goalkeeper",
                                            "es": "en jugada iniciada por el portero",
                                            "el": "σε φάση που ξεκίνησε ο τερματοφύλακας"},
    "from a corner": {"en": "from a corner", "es": "de córner", "el": "από κόρνερ"},
    "from another kind of play": {"en": "from another kind of play", "es": "en otro tipo de jugada",
                                  "el": "από άλλο είδος φάσης"},
    # pressure
    "under pressure": {"en": "under pressure", "es": "bajo presión", "el": "υπό πίεση"},
    "with no pressure on the passer": {"en": "without pressure", "es": "sin presión", "el": "χωρίς πίεση"},
}

# Labels dropped before the model sees the drivers: default conditions, and two labels too vague to narrate.
NOT_NARRATED: frozenset[str] = frozenset({
    "with the foot", "in open play", "with no pressure on the passer",
    "with another part of the body", "from another kind of play",
})


# Label pairs that contradict each other in one line (a header cannot be along the ground). Review with
# scripts/driver_pairs.py before adding to it.
INCOMPATIBLE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"with a header", "along the ground"}),
    frozenset({"with a header", "played low"}),
    frozenset({"from the goalkeeper's hands", "with a header"}),
})

# Greek adjective forms of the height labels, agreeing with "η πάσα" (a second Greek suggestion in hints mode).
GREEK_ADJECTIVES: dict[str, str] = {"along the ground": "συρτή", "played low": "χαμηλή", "lofted high": "ψηλή"}

# Hints mode, English heights: an adjective form (before "pass") and the post-noun form.
ENGLISH_HEIGHT_HINTS: dict[str, list[str]] = {
    "lofted high": ["lofted", "lofted high"],
    "played low": ["low", "played low"],
    "along the ground": ["ground", "along the ground"],
}

# Hints mode: a second Greek suggestion for two depth zones (after the DEPTH_PHRASES one).
GREEK_DEPTH_HINTS: dict[str, str] = {"defensive_third": "από την άμυνα", "middle_third": "από το κέντρο"}


def position_phrase(depth: str, width: str) -> Phrase:
    """The composed position phrase per language: "<depth phrase>, <width phrase>"."""
    return {lang: f"{DEPTH_PHRASES[depth][lang]}, {WIDTH_PHRASES[width][lang]}" for lang in DEPTH_PHRASES[depth]}


def is_narrated(driver: Driver) -> bool:
    return driver.label not in NOT_NARRATED


def narrated(drivers: list[Driver]) -> list[Driver]:
    """The drivers the model is shown, in rank order with their original ranks: NOT_NARRATED labels left out,
    then, for each INCOMPATIBLE_PAIRS pair, the lower-ranked driver left out. Drivers are taken in rank order
    and one is kept only if it clashes with no driver already kept, so a chain resolves deterministically."""
    kept: list[Driver] = []
    for d in sorted((d for d in drivers if is_narrated(d)), key=lambda d: d.rank):
        if not any(frozenset({d.label, k.label}) in INCOMPATIBLE_PAIRS for k in kept):
            kept.append(d)
    return kept


def suggestions(driver: Driver) -> dict:
    """Hints-mode suggestions. A position driver: {"depth": {lang: [...]}, "width": {lang: [...]}}, the two
    pieces (the Greek depth with its second hint for two zones). Any other driver: {lang: [...]}, the label's
    phrase, with the English adjective / post-noun pair and the Greek adjective form for a height label."""
    if driver.feature == "position":
        depth, width = str(driver.value).split("|")
        depth_hints = {lang: [p] for lang, p in DEPTH_PHRASES[depth].items()}
        if depth in GREEK_DEPTH_HINTS:
            depth_hints["el"].append(GREEK_DEPTH_HINTS[depth])
        return {"depth": depth_hints, "width": {lang: [p] for lang, p in WIDTH_PHRASES[width].items()}}
    out = {lang: [phrase] for lang, phrase in LABEL_PHRASES[driver.label].items()}
    if driver.label in ENGLISH_HEIGHT_HINTS:
        out["en"] = list(ENGLISH_HEIGHT_HINTS[driver.label])
    if driver.label in GREEK_ADJECTIVES:
        out["el"].append(GREEK_ADJECTIVES[driver.label])
    return out


def driver_phrases(driver: Driver) -> dict[str, Phrase]:
    """{"phrase": {en, es, el}}, plus "depth" and "width" pieces for a position driver. KeyError if unknown."""
    if driver.feature == "position":
        depth, width = str(driver.value).split("|")
        return {"phrase": position_phrase(depth, width), "depth": DEPTH_PHRASES[depth], "width": WIDTH_PHRASES[width]}
    return {"phrase": LABEL_PHRASES[driver.label]}
