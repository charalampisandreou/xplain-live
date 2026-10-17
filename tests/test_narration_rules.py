import json
import re
from collections import Counter

import pytest

from xplain.narration import NarratorOut, difficulty_tier
from xplain.narrator import FRAME_DRIVERS, FRAMES, OUTCOMES, build_facts, frame_for, variation
from xplain.phrases import (
    ENGLISH_HEIGHT_HINTS,
    GREEK_ADJECTIVES,
    GREEK_DEPTH_HINTS,
    INCOMPATIBLE_PAIRS,
    narrated,
    suggestions,
)
from xplain.prompts import EXAMPLES, FRAME_SKETCHES, INSTRUCTIONS, render_examples
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
    assert result.reasons == []
    # F1 and S2 (concessions) carry no difficulty word: recommended, so only a note
    assert all(n.startswith("difficulty: ") and n.endswith("has no difficulty word") for n in result.notes)
    assert bool(result.notes) == (ex.name in ("F1", "S2"))


@pytest.mark.parametrize("ex", EXAMPLES, ids=[ex.name for ex in EXAMPLES])
def test_example_facts_match_what_the_model_sees_in_free_mode(ex):
    facts = build_facts(example_bundle(ex), "free")   # the example shows the nudge it follows, not the hashed one
    keys = ("kind", "player", "completed", "difficulty_word", "drivers")
    assert {k: facts[k] for k in keys} == {k: ex.facts()[k] for k in keys}
    assert set(ex.facts()) == set(facts)


def test_each_kind_has_both_roles_in_the_examples():
    frames = {kind: {ex.variation[0] for ex in EXAMPLES if ex.kind == kind}
              for kind in ("surprise_failure", "surprise_success")}
    assert all(f == {"reason", "concession"} for f in frames.values()), frames
    for ex in EXAMPLES:
        assert ex.variation[2] in OUTCOMES[ex.kind] and ex.variation[0] in FRAMES
        assert ex.variation[1] == len(ex.drivers_used)


def test_s1_keeps_its_xp_and_takes_the_difficult_tier_word():
    s1 = next(ex for ex in EXAMPLES if ex.name == "S1")
    assert s1.xp == 0.25 and s1.tier().el == "δύσκολη"
    assert s1.line_el.startswith("Η πάσα του Ivo Varga ήταν δύσκολη λόγω του ύψους της")


def test_examples_are_in_every_brief_and_say_they_are_not_wording_to_reuse():
    for text in INSTRUCTIONS.values():
        assert "=== EXAMPLES (they show tone and shape, not wording to reuse) ===" in text
        assert render_examples() in text
        assert "placeholder" not in text
    assert [ex.player for ex in EXAMPLES] == ["Ivo Varga", "Teo Marlow", "Mika Dovan", "Roan Delk"]
    assert "Η πάσα του Roan Delk ήταν εύκολη διότι έγινε από το αμυντικό τρίτο, αλλά χάθηκε." in render_examples()


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
        {"rank": 2, "topic": "pitch position", "direction": "raises",   # the two pieces apart, no single label
         "pieces": {"depth": {"label": "from the middle third",
                              "suggested": {"en": ["in the middle third", "in midfield"],
                                            "es": ["en el tercio medio", "en el centro del campo"],
                                            "el": ["στο κέντρο", "από το κέντρο"]}},
                    "width": {"label": "out wide",
                              "suggested": {"en": ["out wide"], "es": ["desde la banda"], "el": ["από τα πλάγια"]}}},
         "use": "one piece only: depth or width, never both"},
    ]
    assert build_facts(make_bundle(), "free")["drivers"][1]["label"] == "from the middle third, out wide"
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


def test_hints_mode_needs_a_listed_wording_free_mode_does_not():
    natural = line_out(line_es="El pase fácil de Weso Wendez se pierde, aunque iba por el suelo.", drivers_used=[1])
    assert verify(natural, make_bundle(), "free").ok
    assert [r.split(":")[0] for r in verify(natural, make_bundle(), "hints").reasons] == ["driver_expressed"]
    assert not verify(natural, make_bundle(), "strict").ok


def test_hints_brief():
    text = INSTRUCTIONS["hints"]
    assert "suggested shows approved, natural wording" in text
    assert "with one of its suggested wordings, in each language, word for word" in text
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
    out = line_out(line_en="Weso Wendez's easy pass goes astray, even though it was along the ground.",
                   drivers_used=[1, 2])
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
    # two drivers pointing the tier's way, so one or two may be asked for
    direction = "lowers" if completed else "raises"
    drivers = [make_driver(1, *WIDE, direction), make_driver(2, *PRESSED, direction)]
    base = make_bundle(xp=0.25 if completed else 0.94, completed=completed, drivers=drivers)
    nudges = [build_facts(_moment(base, f"moment:m{i % 7}-{i:05d}"), "hints")["variation"] for i in range(200)]
    frames = Counter(n["frame"] for n in nudges)
    outcomes = Counter(n["outcome"]["en"] for n in nudges)
    assert set(frames) == set(FRAMES) and min(frames.values()) >= 200 / len(FRAMES) / 2
    assert set(outcomes) == set(OUTCOMES[base.moment.kind])
    assert min(outcomes.values()) >= 200 / len(OUTCOMES[base.moment.kind]) / 2
    assert {n["situation_facts"] for n in nudges if FRAME_DRIVERS[n["frame"]] == 2} == {1, 2}
    assert {n["situation_facts"] for n in nudges if FRAME_DRIVERS[n["frame"]] == 1} == {1}   # cap: length


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
    # an easy pass whose position fact lowers the rating: it may not be mentioned
    drivers = [make_driver(1, *GROUND), make_driver(2, *WIDE, "lowers")]
    for i in range(100):
        nudge = build_facts(_moment(make_bundle(drivers=drivers), f"moment:m1-{i:05d}"), "hints")["variation"]
        assert nudge["situation_facts"] == 1
    # no driver pointing the tier's way: 0 facts, whatever the shape
    against = [make_driver(1, *GROUND, "lowers")]
    nudge = build_facts(_moment(make_bundle(drivers=against), "moment:m1-00001"), "hints")["variation"]
    assert nudge["situation_facts"] == 0 and nudge["frame"] in FRAMES


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
    assert defensive["depth"]["el"] == ["από την άμυνα"]   # the owner's wording for the defensive third
    assert defensive["width"]["el"] == ["από τον άξονα"]
    assert suggestions(make_driver(1, *WIDE))["depth"]["el"] == ["στο επιθετικό τρίτο"]


@pytest.mark.parametrize("lang, phrase", [
    *(("en", p) for hints in ENGLISH_HEIGHT_HINTS.values() for p in hints),
    *(("el", p) for p in GREEK_DEPTH_HINTS.values()),
])
def test_no_banned_pattern_fires_on_a_hint(lang, phrase):
    assert not [p for p in BANNED[lang] if re.search(p, phrase, re.IGNORECASE)]


def test_reason_frames_ask_for_one_driver_and_concession_frames_for_up_to_two():
    assert {f for f, n in FRAME_DRIVERS.items() if n == 1} == {"reason", "tier_first"}
    assert {f for f, n in FRAME_DRIVERS.items() if n == 2} == {"concession", "concession_first", "player_first"}
    drivers = [make_driver(1, *GROUND), make_driver(2, *PRESSED, "raises")]
    nudges = [build_facts(_moment(make_bundle(drivers=drivers), f"moment:m3-{i:05d}"), "hints")["variation"]
              for i in range(200)]
    assert all(n["situation_facts"] == 1 for n in nudges if FRAME_DRIVERS[n["frame"]] == 1)
    assert {n["situation_facts"] for n in nudges if FRAME_DRIVERS[n["frame"]] == 2} == {1, 2}
    one = [make_driver(1, *GROUND)]   # the concession cap still holds: never more than the usable drivers
    for i in range(100):
        nudge = build_facts(_moment(make_bundle(drivers=one), f"moment:m3-{i:05d}"), "hints")["variation"]
        assert nudge["situation_facts"] == 1


def test_brief_asks_for_the_same_reason_in_every_language_and_short_lines():
    for text in INSTRUCTIONS.values():
        assert "All three lines carry the same shape and the same reason" in text
        assert "line_es and line_el give that reason too, not just a description" in text
        assert '"mainly" and "also" are optional; leave them out when the line is long' in text
        assert "Every word counts toward\n  the limit." in text


# --- frames: rotation through a match -----------------------------------------------------------

def test_frames_rotate_through_the_match_by_slot():
    base = make_bundle()
    k = len(FRAMES)
    assert k == 5 and "driver_first" not in FRAMES
    seen = [frame_for(_moment(base, f"moment:m1-{i:05d}"), slot=i) for i in range(2 * k)]
    assert all(a != b for a, b in zip(seen, seen[1:]))          # neighbours never share a frame
    assert set(seen[:k]) == set(FRAMES)                        # every frame once in five moments
    assert seen[:k] == seen[k:]                                # the same order again
    # the slot decides, not the moment: the same slot gives the same frame on a re-run
    assert frame_for(_moment(base, "moment:m1-99999"), slot=3) == seen[3]
    # no slot: the moment_id hash, as before
    assert frame_for(base) == FRAMES[__import__("hashlib").sha256(base.moment.moment_id.encode()).digest()[0] % k]


def test_outcomes_and_openings_rotate_by_slot():
    base = make_bundle()   # a failure
    nudges = [variation(_moment(base, f"moment:m1-{i:05d}"), narrated(base.explanation.drivers), i)
              for i in range(10)]
    outcomes = [n["outcome"]["en"] for n in nudges]
    assert all(a != b for a, b in zip(outcomes, outcomes[1:]))   # neighbours never share an outcome phrase
    assert set(outcomes) == set(OUTCOMES["surprise_failure"])
    assert all(set(n["outcome"]) == {"en", "es", "el"} for n in nudges)
    assert nudges[1]["outcome"] == {"en": "is lost", "es": "se pierde", "el": "χάνεται"}
    # an opening only for the reason and concession frames; it alternates from one appearance to the next
    for n in nudges:
        assert ("opening" in n) == (n["frame"] in ("reason", "concession"))
    reason = [n["opening"]["en"] for n in nudges if n["frame"] == "reason"]
    assert reason == ["Weso Wendez's pass", "The pass from Weso Wendez"] or \
        reason == ["The pass from Weso Wendez", "Weso Wendez's pass"]
    greek = {n["opening"]["el"] for n in nudges if "opening" in n}
    assert greek == {"Η πάσα του Weso Wendez", "Η μπαλιά του Weso Wendez"}


def test_the_slot_reaches_the_facts():
    base = make_bundle()
    for slot in range(6):
        assert build_facts(base, "hints", slot)["variation"]["frame"] == frame_for(base, slot)
    assert build_facts(base, "hints")["variation"] == variation(base, narrated(base.explanation.drivers))


def test_every_frame_has_a_sketch_in_the_free_and_hints_briefs_only():
    assert tuple(FRAME_SKETCHES) == FRAMES
    for mode in ("free", "hints"):
        for name, sketch in FRAME_SKETCHES.items():
            assert f"- {name}: {sketch}" in INSTRUCTIONS[mode]
    assert "Frames (variation.frame)" not in INSTRUCTIONS["strict"]


def test_player_first_sketch_is_neutral_about_the_result():
    sketch = FRAME_SKETCHES["player_first"]
    assert "plays an easy pass that does not find a teammate" in sketch
    assert "never wording that blames the player (gives the ball away)" in sketch
    assert sketch.count("gives the ball away") == 1   # only in the rule that forbids it


def test_greek_rule_keeps_the_reason():
    for mode in ("free", "hints"):
        assert "never only before πάσα as a description" in INSTRUCTIONS[mode]
        assert "prefer an adjective before πάσα" not in INSTRUCTIONS[mode]
