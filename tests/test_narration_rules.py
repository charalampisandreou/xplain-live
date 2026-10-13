import json
import re
from collections import Counter

import pytest

from xplain.narration import NarratorOut, difficulty_tier
from xplain.narrator import OUTCOMES, SHAPES, build_facts, variation
from xplain.phrases import (
    ENGLISH_HEIGHT_HINTS,
    GREEK_ADJECTIVES,
    GREEK_DEPTH_HINTS,
    INCOMPATIBLE_PAIRS,
    narrated,
    suggestions,
)
from xplain.prompts import EXAMPLES, INSTRUCTIONS, render_examples
from xplain.verifier import BANNED, LANGS, verify

from .test_verifier import line_out, make_bundle, make_driver, reasons_for

# --- the worked examples ------------------------------------------------------------------------


def example_bundle(ex):
    drivers = [make_driver(i, *spec) for i, spec in enumerate(ex.drivers, start=1)]
    return make_bundle(xp=ex.xp, completed=ex.kind == "surprise_success", drivers=drivers, player=ex.player)


def example_out(ex) -> NarratorOut:
    return NarratorOut(line_en=ex.line_en, line_es=ex.line_es, line_el=ex.line_el, drivers_used=list(ex.drivers_used))


@pytest.mark.parametrize("mode", ["free", "hints"])
@pytest.mark.parametrize("ex", EXAMPLES, ids=[ex.name for ex in EXAMPLES])
def test_every_example_passes_the_verifier(ex, mode):
    bundle = example_bundle(ex)
    assert difficulty_tier(bundle.moment) == ex.tier()
    result = verify(example_out(ex), bundle, mode)
    assert (result.reasons, result.notes) == ([], [])   # every line also carries the difficulty idea


@pytest.mark.parametrize("ex", EXAMPLES, ids=[ex.name for ex in EXAMPLES])
def test_example_facts_match_what_the_model_sees_in_free_mode(ex):
    facts = build_facts(example_bundle(ex), "free")   # the example shows the nudge it follows, not the hashed one
    keys = ("kind", "player", "completed", "difficulty_word", "drivers")
    assert {k: facts[k] for k in keys} == {k: ex.facts()[k] for k in keys}
    assert set(ex.facts()) == set(facts)


def test_each_kind_has_two_shapes_in_the_examples():
    shapes = {kind: {ex.variation[0] for ex in EXAMPLES if ex.kind == kind}
              for kind in ("surprise_failure", "surprise_success")}
    assert all(len(s) >= 2 for s in shapes.values()), shapes
    for ex in EXAMPLES:
        assert ex.variation[2] in OUTCOMES[ex.kind] and ex.variation[0] in SHAPES
        assert ex.variation[1] == len(ex.drivers_used)


def test_examples_are_in_every_brief_and_say_they_are_not_wording_to_reuse():
    for text in INSTRUCTIONS.values():
        assert "=== EXAMPLES (they show tone and shape, not wording to reuse) ===" in text
        assert render_examples() in text
        assert "placeholder" not in text
    assert [ex.player for ex in EXAMPLES] == ["Teo Marlow", "Dario Quill", "Pelo Hart", "Ivo Varga", "Roan Delk",
                                              "Mika Dovan"]
    assert "Από την άμυνα, χάνεται η εύκολη πάσα του Roan Delk." in render_examples()


@pytest.mark.parametrize("lang, line", [(lang, getattr(ex, f"line_{lang}")) for ex in EXAMPLES for lang in LANGS])
def test_no_banned_pattern_fires_on_an_example_line(lang, line):
    assert not [p for p in BANNED[lang] if re.search(p, line, re.IGNORECASE)]


# --- banned intent / effort / credit verbs ------------------------------------------------------

@pytest.mark.parametrize("lang, line, word", [
    ("en", "Weso Wendez tries an easy pass that goes astray.", "tries"),
    ("en", "Weso Wendez, trying an easy pass, sees it go astray.", "trying"),
    ("en", "Weso Wendez attempts an easy pass that goes astray.", "attempts"),
    ("en", "Weso Wendez's attempted easy pass goes astray.", "attempted"),
    ("en", "Weso Wendez pulls off a difficult pass.", "pulls off"),
    ("en", "Weso Wendez threads a difficult pass to a teammate.", "threads"),
    ("en", "Weso Wendez manages to complete a difficult pass.", "manages to"),
    ("es", "Weso Wendez intenta un pase fácil que se pierde.", "intenta"),
    ("es", "Weso Wendez trata de dar un pase fácil y lo pierde.", "trata de"),
    ("es", "Weso Wendez logra completar un pase difícil.", "logra"),
    ("el", "Ο Weso Wendez επιχειρεί μια εύκολη πάσα που χάνεται.", "επιχειρεί"),
    ("el", "Ο Weso Wendez προσπαθεί μια εύκολη πάσα που χάνεται.", "προσπαθεί"),
    ("el", "Ο Weso Wendez καταφέρνει μια δύσκολη πάσα.", "καταφέρνει"),
    ("el", "Ο Weso Wendez κατάφερε μια δύσκολη πάσα.", "κατάφερε"),
])
def test_intent_verbs_are_banned(lang, line, word):
    assert [r for r in reasons_for(line_out(**{f"line_{lang}": line})) if r.startswith("banned:")] == [
        f"banned: line_{lang} says {word!r}"]


@pytest.mark.parametrize("lang, line", [
    ("en", "Weso Wendez's easy pass goes astray near the entry to midfield."),   # "try" inside "entry"
    ("en", "Weso Wendez reattempts nothing and his easy pass goes astray."),     # "attempt" inside another word
    ("en", "Weso Wendez pulls back an easy pass that goes astray."),
    ("en", "Weso Wendez's easy pass goes astray from a threadbare move."),
    ("en", "Weso Wendez manages the ball but his easy pass goes astray."),
    ("es", "Se trata de un pase fácil de Weso Wendez que se pierde."),
    ("es", "Bajo una presión intensa, el pase fácil de Weso Wendez se pierde."),
    ("el", "Ο επιχειρηματίας Weso Wendez χάνει μια εύκολη πάσα."),
    ("el", "Η εύκολη πάσα του Weso Wendez φέρνει μόνο απογοήτευση."),
])
def test_intent_verb_near_misses_do_not_fire(lang, line):
    assert not [r for r in reasons_for(line_out(**{f"line_{lang}": line})) if r.startswith("banned:")]


# --- hints mode ---------------------------------------------------------------------------------

def test_hints_facts_carry_label_and_suggestions_for_narrated_drivers_only():
    facts = build_facts(make_bundle(minute=45, added=2), "hints")
    assert facts["drivers"] == [   # driver 3 ("with no pressure on the passer") is not narrated
        {"rank": 1, "topic": "pass height", "direction": "raises", "label": "along the ground",
         "suggested": {"en": ["ground", "along the ground"], "es": ["raso"], "el": ["συρτά", "συρτή"]}},
        {"rank": 2, "topic": "pitch position", "direction": "raises", "label": "from the middle third, out wide",
         "suggested": {"depth": {"en": ["in the middle third"], "es": ["en el tercio medio"],
                                 "el": ["στο κέντρο", "από το κέντρο"]},
                       "width": {"en": ["out wide"], "es": ["desde la banda"], "el": ["από τα πλάγια"]}}},
    ]
    assert "συρτά" not in json.dumps(build_facts(make_bundle(n_drivers=0), "hints"), ensure_ascii=False)


def test_suggestions_are_absent_in_free_mode_and_strict_is_unchanged():
    bundle = make_bundle()
    assert all("suggested" not in d for d in build_facts(bundle, "free")["drivers"])
    strict = build_facts(bundle, "strict")["drivers"]
    assert all("suggested" not in d and "label" not in d and "phrase" in d for d in strict)


def test_greek_adjectives_only_for_the_three_heights():
    assert set(GREEK_ADJECTIVES) == {"along the ground", "played low", "lofted high"}
    header = make_driver(1, "body_part", "body part", "Head", "with a header")
    assert suggestions(header) == {"en": ["with a header"], "es": ["de cabeza"], "el": ["με κεφαλιά"]}


def test_hints_mode_verifies_like_free_mode():
    natural = line_out(line_es="El pase fácil por el suelo de Weso Wendez se pierde.", drivers_used=[1])
    assert verify(natural, make_bundle(), "hints").ok and verify(natural, make_bundle(), "free").ok
    assert not verify(natural, make_bundle(), "strict").ok


def test_hints_brief():
    text = INSTRUCTIONS["hints"]
    assert "suggested shows approved, natural wording" in text
    assert "Never use both pieces of a pitch position suggestion in one line" in text
    assert "verbatim" not in text and "suggested" not in INSTRUCTIONS["free"]


# --- incompatible driver pairs ------------------------------------------------------------------

HEADER = ("body_part", "body part", "Head", "with a header")
GROUND = ("height", "pass height", "Ground Pass", "along the ground")
LOW = ("height", "pass height", "Low Pass", "played low")
HANDS = ("body_part", "body part", "Keeper Arm", "from the goalkeeper's hands")
PRESSED = ("under_pressure", "pressure", 1.0, "under pressure")
WIDE = ("position", "pitch position", "attacking_third|out_wide", "from the attacking third, out wide")


def test_the_initial_incompatible_pairs():
    assert INCOMPATIBLE_PAIRS == {frozenset({"with a header", "along the ground"}),
                                  frozenset({"with a header", "played low"}),
                                  frozenset({"from the goalkeeper's hands", "with a header"})}


@pytest.mark.parametrize("mode", ["free", "hints", "strict"])
def test_the_higher_ranked_driver_of_a_pair_survives(mode):
    bundle = make_bundle(drivers=[make_driver(1, *GROUND), make_driver(2, *PRESSED), make_driver(3, *HEADER)])
    assert [d["rank"] for d in build_facts(bundle, mode)["drivers"]] == [1, 2]
    assert [d.label for d in bundle.explanation.drivers][2] == "with a header"   # the Explanation keeps all three
    swapped = make_bundle(drivers=[make_driver(1, *HEADER), make_driver(2, *GROUND)])
    assert [d["rank"] for d in build_facts(swapped, mode)["drivers"]] == [1]


def test_a_pair_that_is_not_listed_is_untouched():
    drivers = [make_driver(1, *HEADER), make_driver(2, *PRESSED), make_driver(3, *WIDE)]
    assert [d.rank for d in narrated(drivers)] == [1, 2, 3]


def test_a_chain_of_three_resolves_in_rank_order():
    # ground(1) clashes with header(2); header(2) clashes with hands(3); ground and hands do not clash.
    # header is dropped because of ground, so hands no longer clashes with anything kept and stays.
    drivers = [make_driver(1, *GROUND), make_driver(2, *HEADER), make_driver(3, *HANDS)]
    assert [d.rank for d in narrated(drivers)] == [1, 3]
    assert [d.rank for d in narrated(list(reversed(drivers)))] == [1, 3]   # input order does not matter
    # header first: both of its partners go
    chain = [make_driver(1, *HEADER), make_driver(2, *LOW), make_driver(3, *HANDS)]
    assert [d.rank for d in narrated(chain)] == [1]


def test_a_dropped_partner_cannot_be_used_in_a_line():
    bundle = make_bundle(drivers=[make_driver(1, *GROUND), make_driver(2, *HEADER)])
    out = line_out(line_en="Weso Wendez's easy pass along the ground goes astray.", drivers_used=[1, 2])
    assert reasons_for(out, bundle) == ["drivers: drivers_used has [2], the narrated drivers are [1]"]


# --- variation nudge ----------------------------------------------------------------------------

def _moment(bundle, moment_id):
    return bundle.model_copy(update={"moment": bundle.moment.model_copy(update={"moment_id": moment_id})})


def test_the_same_moment_always_gets_the_same_nudge():
    a = build_facts(_moment(make_bundle(), "moment:m7-00123"), "hints")["variation"]
    b = build_facts(_moment(make_bundle(), "moment:m7-00123"), "free")["variation"]
    assert a == b == variation(_moment(make_bundle(), "moment:m7-00123"),
                               narrated(make_bundle().explanation.drivers))
    assert "variation" not in build_facts(make_bundle(), "strict")


@pytest.mark.parametrize("completed", [False, True])
def test_nudges_are_spread_across_moments(completed):
    # drivers that allow every shape: a position fact and a second fact, both pointing the tier's way
    direction = "lowers" if completed else "raises"
    drivers = [make_driver(1, *WIDE, direction), make_driver(2, *PRESSED, direction)]
    base = make_bundle(xp=0.25 if completed else 0.94, completed=completed, drivers=drivers)
    nudges = [build_facts(_moment(base, f"moment:m{i % 7}-{i:05d}"), "hints")["variation"] for i in range(200)]
    shapes = Counter(n["shape"] for n in nudges)
    outcomes = Counter(n["outcome"] for n in nudges)
    assert set(shapes) == set(SHAPES) and min(shapes.values()) >= 25          # 50 each if even
    assert set(outcomes) == set(OUTCOMES[base.moment.kind])
    assert min(outcomes.values()) >= 200 / len(OUTCOMES[base.moment.kind]) / 2
    two = [n for n in nudges if n["shape"] != "short"]
    assert {n["situation_facts"] for n in two} == {1, 2}
    assert all(n["situation_facts"] == 1 for n in nudges if n["shape"] == "short")


@pytest.mark.parametrize("drivers, at_most", [
    ([], 0),
    ([GROUND], 1),
    ([GROUND, PRESSED], 2),
    ([GROUND, ("position", "pitch position", "middle_third|out_wide", "from the middle third, out wide")], 2),
])
def test_the_nudge_never_asks_for_more_drivers_than_are_narrated(drivers, at_most):
    built = [make_driver(i, *spec) for i, spec in enumerate(drivers, start=1)]
    for i in range(100):
        nudge = build_facts(_moment(make_bundle(drivers=built), f"moment:m1-{i:05d}"), "hints")["variation"]
        assert nudge["situation_facts"] <= at_most


def test_the_nudge_counts_only_drivers_that_match_the_tier():
    # an easy pass whose only position fact lowers the rating: it may not be mentioned, so no situation_first
    drivers = [make_driver(1, *GROUND), make_driver(2, *WIDE, "lowers")]
    for i in range(100):
        nudge = build_facts(_moment(make_bundle(drivers=drivers), f"moment:m1-{i:05d}"), "hints")["variation"]
        assert nudge["situation_facts"] <= 1 and nudge["shape"] != "situation_first"


def test_free_and_hints_briefs_explain_the_nudge_and_strict_does_not():
    for mode in ("free", "hints"):
        assert "Follow the variation nudge unless it makes the sentence unnatural" in INSTRUCTIONS[mode]
        assert "never the facts" in INSTRUCTIONS[mode]
    assert "Follow the variation nudge" not in INSTRUCTIONS["strict"]
    assert "player, team" not in INSTRUCTIONS["hints"]


# --- hints wording ------------------------------------------------------------------------------

@pytest.mark.parametrize("spec, en", [
    (("height", "pass height", "High Pass", "lofted high"), ["lofted", "lofted high"]),
    (("height", "pass height", "Low Pass", "played low"), ["low", "played low"]),
    (GROUND, ["ground", "along the ground"]),
])
def test_english_height_hints_have_an_adjective_and_a_post_noun_form(spec, en):
    s = suggestions(make_driver(1, *spec))
    assert s["en"] == en and len(s["es"]) == 1 and len(s["el"]) == 2


def test_greek_depth_hints():
    defensive = suggestions(make_driver(1, "position", "pitch position", "defensive_third|central",
                                        "from their own defensive third, centrally"))
    assert defensive["depth"]["el"] == ["στην άμυνα", "από την άμυνα"]
    assert defensive["width"]["el"] == ["από τον άξονα"]
    assert suggestions(make_driver(1, *WIDE))["depth"]["el"] == ["στο επιθετικό τρίτο"]


@pytest.mark.parametrize("lang, phrase", [
    *(("en", p) for hints in ENGLISH_HEIGHT_HINTS.values() for p in hints),
    *(("el", p) for p in GREEK_DEPTH_HINTS.values()),
])
def test_no_banned_pattern_fires_on_a_hint(lang, phrase):
    assert not [p for p in BANNED[lang] if re.search(p, phrase, re.IGNORECASE)]
