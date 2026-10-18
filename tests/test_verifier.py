import re

import pytest
from pydantic import ValidationError

from xplain import analytics
from xplain.analyst import build_evidence
from xplain.clock import clock_label
from xplain.contracts import Driver, Explanation, Moment, MomentBundle, PassGeometry
from xplain.generator import generate_match
from xplain.narration import DIFFICULTY_TIERS, NarratorOut, difficulty_tier
from xplain.pipeline import process_match
from xplain.scoring import ModelLoadError, XPassScorer, find_default_model
from xplain.phrases import driver_phrases
from xplain.verifier import (EASY_WORDS, HARD_WORDS, LANGS, MAX_WORDS, TABLE_PHRASES, difficulty_words, line_roles,
                             narrated_drivers, verify)

LABELS = ("along the ground", "from the middle third, out wide", "with no pressure on the passer")
# (feature, topic, value, label) for the default drivers: rank 3 is a default condition the model never sees
DRIVER_SPECS = (
    ("height", "pass height", "Ground Pass", LABELS[0]),
    ("position", "pitch position", "middle_third|out_wide", LABELS[1]),
    ("under_pressure", "pressure", 0.0, LABELS[2]),
)


def make_driver(rank: int, feature: str, topic: str, value, label: str, direction: str = "raises") -> Driver:
    return Driver(rank=rank, feature=feature, topic=topic, value=value, label=label,
                  shap=0.5 if direction == "raises" else -0.5, direction=direction)


def make_bundle(xp: float = 0.94, minute: int = 3, n_drivers: int = 3, completed: bool = False,
                added: int = 0, drivers: list[Driver] | None = None, player: str = "Weso Wendez") -> MomentBundle:
    """A small Moment + Explanation built by hand, with Analyst-style evidence."""
    event_id = "m1-00042"
    d = analytics.difficulty(xp)
    dlt = analytics.delta(completed, xp)
    evidence = build_evidence(event_id, {"xp": xp, "delta": dlt, "difficulty": d, "minute": minute,
                                         "minute_added": added, "minute_label": clock_label(minute, added),
                                         "completed": completed, "player_name": player})
    moment = Moment(
        moment_id=f"moment:{event_id}", kind="surprise_success" if completed else "surprise_failure",
        event_id=event_id, match_id="m1", seq=42, t=(minute - 1) * 60.0 + 5, minute=minute, minute_added=added,
        minute_label=clock_label(minute, added), period=1,
        team_id="home", player_id="p7", player_name=player, completed=completed, xp=xp, delta=dlt,
        difficulty=d, band=analytics.difficulty_band(d), significance=abs(dlt),
        pass_geometry=PassGeometry(start_x=50, start_y=75, end_x=70, end_y=70, length=20.6, ball_speed_ms=14),
        evidence=evidence, model_version="test",
    )
    if drivers is None:
        drivers = [make_driver(i + 1, *DRIVER_SPECS[i]) for i in range(n_drivers)]
    explanation = Explanation(
        moment_id=moment.moment_id, event_id=event_id, method="test", unit="log-odds", baseline=0.0,
        raw_score=sum(d.shap for d in drivers), drivers=drivers, rest=0.0, evidence={}, model_version="test",
    )
    return MomentBundle(moment=moment, explanation=explanation, qualifies_live=False, qualified_by=())


def hard_bundle(**kw) -> MomentBundle:
    """A completed pass the model rated hard (surprise_success, xP 0.25: tier "difficult"); its default drivers
    point to hard ("lowers"), as the direction check requires."""
    if "drivers" not in kw:
        kw["drivers"] = [make_driver(i + 1, *DRIVER_SPECS[i], "lowers") for i in range(kw.pop("n_drivers", 3))]
    return make_bundle(xp=0.25, completed=True, **kw)


def make_out(**kw) -> NarratorOut:
    """A clean line for make_bundle(): an easy pass that goes astray, with driver 1 (along the ground) conceded in
    every language."""
    base = dict(
        line_en="Weso Wendez's easy pass goes astray, even though it was along the ground.",
        line_es="El pase fácil de Weso Wendez se pierde, aunque iba raso.",
        line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που πήγε συρτά.",
        drivers_used=[1],
    )
    return NarratorOut(**{**base, **kw})


def line_out(**kw) -> NarratorOut:
    """make_out() for tests of other checks. It cites driver 1 (along the ground), which every line names as a
    concession, so needs_driver, unsupported_fact, driver_expressed and role_parity stay quiet on make_bundle()."""
    return make_out(**{"drivers_used": [1], **kw})


def bare_out(**kw) -> NarratorOut:
    """A line that names no driver and cites none, for moments with no driver the line could use."""
    base = dict(line_en="Weso Wendez's easy ball goes astray.", line_es="El pase fácil de Weso Wendez se pierde.",
                line_el="Η εύκολη πάσα του Weso Wendez χάνεται.", drivers_used=[])
    return NarratorOut(**{**base, **kw})


def plain_reasons(lang: str, line: str) -> list[str]:
    """The reasons for one line of bare_out() on a moment with no narrated driver: for checks of the line itself
    (numbers, length, sentence, banned words), so no driver check speaks."""
    return reasons_for(bare_out(**{f"line_{lang}": line}), make_bundle(n_drivers=0))


HARD_OUT = dict(
    line_en="Weso Wendez's pass was difficult because it was along the ground, but it finds a teammate.",
    line_es="El pase de Weso Wendez era difícil porque iba raso, pero encuentra a un compañero.",
    line_el="Η πάσα του Weso Wendez ήταν δύσκολη επειδή πήγε συρτά, αλλά βρίσκει συμπαίκτη.",
    drivers_used=[1],
)


def reasons_for(out: NarratorOut, bundle: MomentBundle | None = None, mode: str | None = None) -> list[str]:
    return verify(out, bundle or make_bundle(), mode).reasons


# --- contract -----------------------------------------------------------------------------------

def test_narrator_out_is_frozen_and_strict():
    out = make_out()
    with pytest.raises(ValidationError):
        out.line_en = "x"
    with pytest.raises(ValidationError):
        NarratorOut(**out.model_dump(), extra="x")
    with pytest.raises(ValidationError):
        NarratorOut(**out.model_dump(), numbers_used=["94"])   # the field is gone


def test_percent_is_round_100x():
    assert analytics.percent(0.94) == 94
    assert analytics.percent(0.125) == 12      # round half to even
    assert analytics.percent(0.145) == 14      # 100 * 0.145 == 14.499999999999998
    assert analytics.difficulty(0.94) == 6


# --- passing ------------------------------------------------------------------------------------

def test_clean_line_passes():
    result = verify(make_out(), make_bundle())
    assert (result.ok, result.reasons, result.notes) == (True, [], [])


def test_clean_hard_line_passes():
    result = verify(NarratorOut(**HARD_OUT), hard_bundle())
    assert (result.ok, result.reasons, result.notes) == (True, [], [])


def test_real_seed42_bundle_passes():
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc})")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path}")
    bundle = process_match(generate_match(42)[0], XPassScorer(model_path, feature_list_path))[0]
    tier = difficulty_tier(bundle.moment)
    name = bundle.moment.player_name
    wanted = "raises" if tier.family == "easy" else "lowers"
    first = next((d for d in narrated_drivers(bundle).values() if d.direction == wanted), None)
    piece = (driver_phrases(first).get("depth") or driver_phrases(first)["phrase"]) if first else {}  # one position piece
    ph = {lang: f" {p}" for lang, p in piece.items()} if first else dict.fromkeys(LANGS, "")
    concede = ({"en": f", even though it was{ph['en']}", "es": f", aunque fue{ph['es']}",
                "el": f", παρόλο που έγινε{ph['el']}"} if first else dict.fromkeys(LANGS, ""))
    out = NarratorOut(
        line_en=f"{name}'s {tier.en} pass goes astray{concede['en']}.",
        line_es=f"El pase {tier.es} de {name} se pierde{concede['es']}.",
        line_el=f"Η {tier.el} πάσα του {name} χάνεται{concede['el']}.",
        drivers_used=[first.rank] if first else [],
    )
    result = verify(out, bundle)
    assert (result.reasons, result.notes) == ([], [])


# --- difficulty tiers ---------------------------------------------------------------------------

@pytest.mark.parametrize("xp, completed, en, family", [
    (0.997, False, "very easy", "easy"), (0.96, False, "very easy", "easy"),
    (0.9599, False, "easy", "easy"), (0.92, False, "easy", "easy"),
    (0.35, True, "tricky", "hard"), (0.30, True, "tricky", "hard"),
    (0.2999, True, "difficult", "hard"), (0.20, True, "difficult", "hard"),
    (0.1999, True, "very difficult", "hard"), (0.0, True, "very difficult", "hard"),
])
def test_difficulty_tier_cutoffs(xp, completed, en, family):
    tier = difficulty_tier(make_bundle(xp=xp, completed=completed).moment)
    assert (tier.en, tier.family) == (en, family)


def test_every_tier_word_matches_its_own_family_only():
    for tiers in DIFFICULTY_TIERS.values():
        for _, tier in tiers:
            for lang, word in tier.words().items():
                found = difficulty_words(lang, word)
                assert found[tier.family], (lang, word)
                other = "hard" if tier.family == "easy" else "easy"
                assert not found[other], (lang, word)


@pytest.mark.parametrize("lang, line", [
    ("en", "Wendez's tough ball goes astray, even though it was along the ground."),
    ("es", "El pase difícil de Wendez se pierde, aunque iba raso."),
    ("el", "Η δύσκολη πάσα του Wendez χάνεται, παρόλο που ήταν συρτή."),
    ("el", "Η περίπλοκη πάσα του Wendez χάνεται, παρόλο που ήταν συρτή."),   # the new tricky word is a hard word
])
def test_hard_word_on_an_easy_pass_is_rejected(lang, line):
    reasons = reasons_for(line_out(**{f"line_{lang}": line}))
    assert len(reasons) == 1
    assert reasons[0].startswith(f"difficulty: line_{lang} says ")
    assert reasons[0].endswith("but the pass is rated easy (easy)")


@pytest.mark.parametrize("lang, line", [
    ("en", "Weso Wendez's pass was difficult because it was along the ground, but the simple ball finds a teammate."),
    ("es", "El pase de Weso Wendez era difícil porque iba raso, pero el balón sencillo encuentra a un compañero."),
    ("el", "Η πάσα του Weso Wendez ήταν δύσκολη επειδή ήταν συρτή, αλλά η εύκολη μπαλιά βρίσκει συμπαίκτη."),
])
def test_easy_word_on_a_hard_pass_is_rejected(lang, line):
    out = NarratorOut(**{**HARD_OUT, "drivers_used": [1], f"line_{lang}": line})
    reasons = reasons_for(out, hard_bundle())
    assert len(reasons) == 1
    assert reasons[0].startswith(f"difficulty: line_{lang} says ")
    assert reasons[0].endswith("but the pass is rated hard (difficult)")


def test_greek_accent_variants_are_found():
    assert difficulty_words("el", "μια ευκολότερη πάσα")["easy"]
    assert difficulty_words("el", "ΔΥΣΚΟΛΗ ΠΑΣΑ")["hard"]


def test_line_without_a_difficulty_word_is_accepted_but_noted():
    out = line_out(line_en="Weso Wendez's pass goes astray, even though it was along the ground.",
                   line_el="Η πάσα του Weso Wendez χάνεται, παρόλο που ήταν συρτή.")
    result = verify(out, make_bundle())
    assert result.ok and result.reasons == []
    assert result.notes == ["difficulty: line_en has no difficulty word", "difficulty: line_el has no difficulty word"]


# --- no numbers ---------------------------------------------------------------------------------

@pytest.mark.parametrize("lang, line, reason", [
    ("en", "Weso Wendez's easy 94 ball goes astray.", "no_numbers: line_en has '94'"),
    ("en", "Weso Wendez's easy % ball goes astray.", "no_numbers: line_en has '%'"),
    ("en", "Weso Wendez's easy 94% ball goes astray.", "no_numbers: line_en has '94%'"),
    ("es", "En el 45+2, el pase fácil de Weso Wendez se pierde.", "no_numbers: line_es has '45'"),
    ("el", "Στο 3ο, η εύκολη πάσα του Weso Wendez χάνεται.", "no_numbers: line_el has '3'"),
])
def test_digits_and_percent_are_rejected(lang, line, reason):
    assert reason in reasons_for(line_out(**{f"line_{lang}": line}))


@pytest.mark.parametrize("lang, line, word", [
    ("en", "Late in the game, Weso Wendez's easy ball goes astray in the final minute.", "minute"),
    ("en", "Weso Wendez's easy ball goes astray with minutes to go.", "minutes"),
    ("en", "Weso Wendez's easy, ninety percent ball goes astray.", "percent"),
    ("en", "Weso Wendez's easy, ninety per cent ball goes astray.", "per cent"),
    ("es", "En el último minuto, el pase fácil de Weso Wendez se pierde.", "minuto"),
    ("es", "En los minutos finales, el pase fácil de Weso Wendez se pierde.", "minutos"),
    ("es", "El pase fácil, noventa por ciento, de Weso Wendez se pierde.", "por ciento"),
    ("el", "Στο τελευταίο λεπτό, η εύκολη πάσα του Weso Wendez χάνεται.", "λεπτό"),
    ("el", "Στα τελευταία λεπτά, η εύκολη πάσα του Weso Wendez χάνεται.", "λεπτά"),
    ("el", "Η εύκολη πάσα του Weso Wendez με υψηλό ποσοστό χάνεται.", "ποσοστό"),
])
def test_minute_and_percent_words_are_rejected(lang, line, word):
    assert plain_reasons(lang, line) == [f"no_numbers: line_{lang} says {word!r}"]


def test_greek_delicate_pass_is_not_a_minute():
    assert plain_reasons("el", "Η λεπτή και εύκολη πάσα του Weso Wendez χάνεται.") == []


def test_spike_lines_with_numbers_are_now_rejected():
    out = NarratorOut(
        line_en="At 3 minutes, Weso Wendez makes a 94% pass attempt from the middle third, out wide, but fails.",
        line_es="A los 3 minutos, Weso Wendez intenta un pase del 94% desde el tercio medio, hacia afuera, pero falla.",
        line_el="Στο 3ο λεπτό, ο Weso Wendez επιχειρεί μια πάσα 94% από το μεσαίο τρίτο, προς τα έξω, αλλά αποτυγχάνει.",
        drivers_used=[2],   # the English line says "out wide", a piece of driver 2
    )
    reasons = reasons_for(out)
    assert {"no_numbers", "banned"} <= {r.split(":")[0] for r in reasons}   # banned: the intent verbs
    assert {r.split(" ")[1] for r in reasons if r.startswith("no_numbers")} == {"line_en", "line_es", "line_el"}
    assert [r for r in reasons if r.startswith("banned")] == [
        "banned: line_en says 'attempt'", "banned: line_es says 'intenta'", "banned: line_el says 'επιχειρεί'"]


# --- length and sentence ------------------------------------------------------------------------

def test_word_limits():
    assert MAX_WORDS == {"en": 24, "es": 24, "el": 24}


@pytest.mark.parametrize("lang", LANGS)
def test_over_length_line_is_rejected(lang):
    n = MAX_WORDS[lang] + 1
    line = " ".join(["word"] * n) + "."
    assert plain_reasons(lang, line) == [f"length: line_{lang} has {n} words, max {MAX_WORDS[lang]}"]


@pytest.mark.parametrize("lang", LANGS)
def test_exactly_max_words_passes(lang):
    line = " ".join(["word"] * MAX_WORDS[lang]) + "."
    assert plain_reasons(lang, line) == []


@pytest.mark.parametrize("line, reason", [
    ("", "sentence: line_en is empty"),
    ("   ", "sentence: line_en is empty"),
    ("An easy ball\ngoes astray.", "sentence: line_en has a line break"),
    ("An easy ball. It goes astray.", "sentence: line_en has more than one sentence"),
])
def test_line_must_be_one_non_empty_sentence(line, reason):
    assert plain_reasons("en", line) == [reason]


def test_greek_question_mark_ends_a_sentence():
    assert plain_reasons("el", "Χάνεται η εύκολη πάσα; Ναι, του Weso Wendez.") == [
        "sentence: line_el has more than one sentence"]


# --- banned claims ------------------------------------------------------------------------------

@pytest.mark.parametrize("field, line", [
    # box / penalty area
    ("line_en", "Wendez misses an easy pass inside the box."),
    ("line_es", "Wendez falla un pase fácil dentro del área."),
    ("line_el", "Ο Wendez χάνει εύκολη πάσα μέσα στην περιοχή."),
    # line-breaking
    ("line_en", "Wendez tries an easy line-breaking pass."),
    ("line_en", "Wendez's easy pass breaks the line."),
    ("line_es", "El pase fácil de Wendez rompe las líneas."),
    ("line_el", "Η εύκολη πάσα του Wendez σπάει τις γραμμές."),
    # defender counts
    ("line_en", "Wendez's easy pass beats three defenders."),
    ("line_es", "El pase fácil de Wendez supera a dos rivales."),
    ("line_el", "Η εύκολη πάσα του Wendez περνά δύο αμυντικούς."),
    # left / right
    ("line_en", "Wendez misses an easy pass on the left."),
    ("line_en", "Wendez misses an easy pass with his right foot."),
    ("line_es", "Wendez falla un pase fácil por la izquierda."),
    ("line_el", "Ο Wendez χάνει εύκολη πάσα από τα δεξιά."),
    # model internals
    ("line_en", "SHAP says the easy pass fails."),
    ("line_en", "The easy pass fails against the baseline."),
    ("line_en", "The raw score says easy and the pass fails."),
    ("line_en", "The expected pass value is high."),
    ("line_es", "La línea base dice fácil."),
    ("line_el", "Η αναμενόμενη αξία είναι υψηλή."),
    # throw
    ("line_en", "Wendez throws away an easy pass."),
    ("line_el", "Ο Wendez ρίχνει μια εύκολη πάσα και χάνει."),
])
def test_banned_claims(field, line):
    reasons = reasons_for(line_out(**{field: line}))
    assert any(r.startswith(f"banned: {field} says ") for r in reasons), reasons


def test_throw_in_is_not_a_throw():
    throw_in = make_driver(2, "play_pattern", "how the move started", "Throw In", "from a throw-in")
    out = line_out(line_en="Wendez's easy pass goes astray, even though it was along the ground and from a throw-in.",
                   line_es="El pase fácil de Wendez se pierde, aunque iba raso y de saque de banda.",
                   line_el="Η εύκολη πάσα του Wendez χάνεται, παρόλο που ήταν συρτή και από πλάγιο άουτ.",
                   drivers_used=[1, 2])
    assert verify(out, make_bundle(drivers=[make_driver(1, *DRIVER_SPECS[0]), throw_in])).ok


def test_spanish_lanza_is_allowed():
    out = line_out(line_es="Wendez lanza un pase fácil y lo pierde, aunque iba raso.")
    assert verify(out, make_bundle()).ok


def test_banned_is_case_insensitive():
    assert plain_reasons("en", "Wendez misses an easy pass INSIDE THE BOX.") == ["banned: line_en says 'INSIDE THE BOX'"]


# --- drivers ------------------------------------------------------------------------------------

def test_driver_rank_that_does_not_exist():
    reasons = reasons_for(make_out(drivers_used=[1, 4]), make_bundle(n_drivers=3))
    assert reasons == ["drivers: drivers_used has [4], the narrated drivers are [1, 2]"]


def test_duplicate_driver_rank():
    assert reasons_for(make_out(drivers_used=[1, 1])) == ["drivers: drivers_used repeats [1]"]


def test_zero_drivers_with_empty_drivers_used_passes():
    assert verify(bare_out(), make_bundle(n_drivers=0)).ok


def test_zero_drivers_with_drivers_used_fails():
    reasons = reasons_for(bare_out(drivers_used=[1]), make_bundle(n_drivers=0))
    assert reasons == ["drivers: the moment has no narrated drivers but drivers_used is [1]"]


# --- several failures ---------------------------------------------------------------------------

def test_two_failures_return_two_reasons():
    out = make_out(line_en="Wendez's easy pass goes astray inside the box, even though it was along the ground.",
                   drivers_used=[1, 5])
    result = verify(out, make_bundle())
    assert not result.ok
    assert result.reasons == [
        "banned: line_en says 'inside the box'",
        "drivers: drivers_used has [5], the narrated drivers are [1, 2]",
    ]


def test_word_lists_compile_for_every_language():
    for table in (EASY_WORDS, HARD_WORDS):
        assert set(table) == set(LANGS)
        for patterns in table.values():
            for p in patterns:
                re.compile(p)


# --- phrases ------------------------------------------------------------------------------------

def test_missing_phrase_is_rejected_per_language_in_strict_mode():
    out = make_out(line_es="El pase fácil de Weso Wendez se pierde, aunque iba por el suelo.")
    assert reasons_for(out, mode="strict") == ["phrases: line_es lists driver 1 but does not say 'raso'"]


def test_free_mode_skips_the_phrases_check():
    out = make_out(line_es="El pase fácil de Weso Wendez se pierde, aunque iba por el suelo.",
                   line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που κύλησε στο χορτάρι.")
    assert verify(out, make_bundle(), "free").ok
    assert [r.split(":")[0] for r in reasons_for(out, mode="strict")] == ["phrases", "phrases"]
    assert [r.split(":")[0] for r in reasons_for(out, mode="hints")] == ["driver_expressed", "driver_expressed"]


@pytest.mark.parametrize("mode", ["free", "strict"])
def test_every_other_check_still_runs_in_both_modes(mode):
    area = make_out(line_es="El pase fácil de Weso Wendez se pierde cerca del área, aunque iba raso.")
    assert reasons_for(area, mode=mode) == ["banned: line_es says 'cerca del área'"]
    vague = line_out(line_en="Weso Wendez's easy ball with another part of the body goes astray, even though it was "
                             "along the ground.")
    assert reasons_for(vague, mode=mode) == ["banned: line_en says 'another part of the body'"]
    assert reasons_for(make_out(drivers_used=[1, 3]), mode=mode) == [
        "drivers: drivers_used has [3], the narrated drivers are [1, 2]"]
    against = NarratorOut(**ONE_EASY, drivers_used=[1])
    assert reasons_for(against, make_bundle(drivers=_two_drivers("lowers", "raises")), mode) == [
        "driver_direction: driver 1 (lofted high) lowers the rating, but the pass is rated easy (easy)"]
    numbers = line_out(line_en="Weso Wendez's easy 94% ball goes astray, even though it was along the ground.")
    assert reasons_for(numbers, mode=mode) == ["no_numbers: line_en has '94%'"]


def test_phrase_match_is_case_insensitive_and_whole_words():
    loud = make_out(line_en="Even though it was ALONG THE GROUND, Weso Wendez's easy ball goes astray.")
    assert verify(loud, make_bundle(), "strict").ok
    out = make_out(line_es="El pase fácil de Weso Wendez acaba en fracaso, aunque iba por el suelo.")   # "raso" in "fracaso"
    assert reasons_for(out, mode="strict") == ["phrases: line_es lists driver 1 but does not say 'raso'"]


POSITION_LINES = {   # driver 2 is "middle_third|out_wide"
    "composed": ("Weso Wendez's easy ball goes astray, even though it was in the middle third, out wide.",
                 "El pase fácil de Weso Wendez se pierde, aunque salió en el tercio medio, desde la banda.",
                 "Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε στο κέντρο, από τα πλάγια."),
    "depth": ("Weso Wendez's easy ball goes astray, even though it was in the middle third.",
              "El pase fácil de Weso Wendez se pierde, aunque salió en el tercio medio.",
              "Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε στο κέντρο."),
    "width": ("Weso Wendez's easy ball goes astray, even though it was out wide.",
              "El pase fácil de Weso Wendez se pierde, aunque salió desde la banda.",
              "Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε από τα πλάγια."),
}


@pytest.mark.parametrize("which", sorted(POSITION_LINES))
def test_position_driver_accepts_the_composed_phrase_or_one_piece(which):
    en, es, el = POSITION_LINES[which]
    assert verify(NarratorOut(line_en=en, line_es=es, line_el=el, drivers_used=[2]), make_bundle(), "strict").ok


def test_position_driver_without_any_piece_is_rejected():
    out = make_out(line_en="Weso Wendez's easy ball goes astray, even though it came from the middle of the pitch.",
                   drivers_used=[2], line_es=POSITION_LINES["width"][1], line_el=POSITION_LINES["depth"][2])
    assert reasons_for(out, mode="strict") == ["phrases: line_en lists driver 2 but does not say "
                                "'in the middle third, out wide' or 'in the middle third' or 'out wide'"]


def test_unused_driver_needs_no_phrase():
    assert verify(make_out(), make_bundle(), "strict").ok   # driver 2 (the position) is not used


# --- driver direction ---------------------------------------------------------------------------

def _two_drivers(direction_1: str, direction_2: str) -> list[Driver]:
    return [make_driver(1, "height", "pass height", "High Pass", "lofted high", direction_1),
            make_driver(2, "under_pressure", "pressure", 1.0, "under pressure", direction_2)]


EASY_TWO = dict(line_en="Weso Wendez's easy pass goes astray, even though it was lofted high and under pressure.",
                line_es="El pase fácil de Weso Wendez se pierde, aunque iba por alto y bajo presión.",
                line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που πήγε ψηλά και υπό πίεση.")
HARD_TWO = dict(line_en="Weso Wendez's difficult pass reaches a teammate, even though it was lofted high and under "
                        "pressure.",
                line_es="El pase difícil de Weso Wendez llega a un compañero, aunque iba por alto y bajo presión.",
                line_el="Η δύσκολη πάσα του Weso Wendez φτάνει σε συμπαίκτη, παρόλο που πήγε ψηλά και υπό πίεση.")


def test_rank_two_driver_against_an_easy_rating_is_rejected():
    out = NarratorOut(**EASY_TWO, drivers_used=[1, 2])
    reasons = reasons_for(out, make_bundle(drivers=_two_drivers("raises", "lowers")))
    assert reasons == ["driver_direction: driver 2 (under pressure) lowers the rating, but the pass is rated easy "
                       "(easy)"]


def test_rank_two_driver_against_a_hard_rating_is_rejected():
    out = NarratorOut(**HARD_TWO, drivers_used=[1, 2])
    reasons = reasons_for(out, hard_bundle(drivers=_two_drivers("lowers", "raises")))
    assert len(reasons) == 1 and reasons[0].startswith("driver_direction: driver 2 (under pressure) raises")


def test_matching_directions_pass():
    assert verify(NarratorOut(**EASY_TWO, drivers_used=[1, 2]), make_bundle(drivers=_two_drivers("raises", "raises"))).ok
    assert verify(NarratorOut(**HARD_TWO, drivers_used=[1, 2]), hard_bundle(drivers=_two_drivers("lowers", "lowers"))).ok


ONE_EASY = dict(line_en="Weso Wendez's easy pass goes astray, even though it was lofted high.",
                line_es="El pase fácil de Weso Wendez se pierde, aunque iba por alto.",
                line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που πήγε ψηλά.")


def test_rank_one_driver_against_the_rating_is_rejected():
    bundle = make_bundle(drivers=_two_drivers("lowers", "raises"))   # easy pass, driver 1 points to hard
    assert reasons_for(NarratorOut(**ONE_EASY, drivers_used=[1]), bundle) == [
        "driver_direction: driver 1 (lofted high) lowers the rating, but the pass is rated easy (easy)"]


def test_rank_one_driver_that_agrees_is_accepted():
    bundle = make_bundle(drivers=_two_drivers("raises", "lowers"))   # driver 2 disagrees but is not used
    assert verify(NarratorOut(**ONE_EASY, drivers_used=[1]), bundle).ok


def test_a_line_with_no_drivers_is_valid_whatever_the_directions():
    bundle = make_bundle(drivers=_two_drivers("lowers", "lowers"))   # easy pass, no driver agrees
    assert verify(bare_out(), bundle).ok


# --- drivers the model never sees ---------------------------------------------------------------

def test_a_dropped_driver_cannot_be_used():
    out = make_out(drivers_used=[1, 3])   # driver 3 is "with no pressure on the passer"
    assert reasons_for(out) == ["drivers: drivers_used has [3], the narrated drivers are [1, 2]"]


def test_all_drivers_dropped_means_no_driver_may_be_used():
    dropped = [make_driver(1, "body_part", "body part", "Foot", "with the foot"),
               make_driver(2, "play_pattern", "how the move started", "Regular Play", "in open play")]
    bundle = make_bundle(drivers=dropped)
    assert verify(bare_out(), bundle).ok
    assert reasons_for(bare_out(drivers_used=[1]), bundle) == [
        "drivers: the moment has no narrated drivers but drivers_used is [1]"]


# --- banned additions: box / area and the two vague labels --------------------------------------

@pytest.mark.parametrize("lang, line, word", [
    ("en", "Weso Wendez's easy ball into the box goes astray.", "the box"),
    ("en", "Weso Wendez's easy ball into the area goes astray.", "the area"),
    ("en", "Weso Wendez's easy ball towards the penalty area goes astray.", "penalty area"),
    ("en", "Weso Wendez's easy ball towards the penalty box goes astray.", "penalty box"),
    ("es", "El pase fácil de Weso Wendez al área se pierde.", "área"),
    ("es", "El pase fácil de Weso Wendez cerca del área se pierde.", "cerca del área"),
    ("es", "El pase fácil de Weso Wendez desde fuera del área se pierde.", "del área"),
    ("el", "Η εύκολη πάσα του Weso Wendez προς την περιοχή χάνεται.", "περιοχή"),
    ("en", "Weso Wendez's easy ball with another part of the body goes astray.", "another part of the body"),
    ("en", "Weso Wendez's easy ball from another kind of play goes astray.", "another kind of play"),
    ("es", "El pase fácil con otra parte del cuerpo de Weso Wendez se pierde.", "otra parte del cuerpo"),
    ("es", "El pase fácil en otro tipo de jugada de Weso Wendez se pierde.", "otro tipo de jugada"),
    ("el", "Η εύκολη πάσα του Weso Wendez με άλλο μέρος του σώματος χάνεται.", "άλλο μέρος του σώματος"),
    ("el", "Η εύκολη πάσα του Weso Wendez από άλλο είδος φάσης χάνεται.", "άλλο είδος φάσης"),
])
def test_banned_additions(lang, line, word):
    assert [r for r in plain_reasons(lang, line) if r.startswith("banned:")] == [f"banned: line_{lang} says {word!r}"]


@pytest.mark.parametrize("lang, line", [
    ("en", "Weso Wendez's easy ball in front of the arena goes astray."),
    ("en", "A boxed-in Weso Wendez sees his easy ball go astray."),
    ("es", "El pase fácil y por vía aérea de Weso Wendez se pierde."),
    ("es", "El pase fácil de Weso Wendez se pierde en la arena."),
    ("el", "Η εύκολη πάσα του Weso Wendez χάνεται και ο ρυθμός περιορίζεται."),
    ("en", "Weso Wendez's easy ball goes astray in another part of the pitch."),
    ("en", "Weso Wendez's easy ball goes astray in another kind of move."),
    ("es", "El pase fácil de Weso Wendez se pierde en otra parte del campo."),
    ("es", "El pase fácil de Weso Wendez es otro tipo de pase y se pierde."),
    ("el", "Η εύκολη πάσα του Weso Wendez χάνεται σε άλλο μέρος του γηπέδου."),
    ("el", "Η εύκολη πάσα του Weso Wendez είναι άλλο είδος πάσας και χάνεται."),
])
def test_banned_near_misses_do_not_fire(lang, line):
    assert not [r for r in plain_reasons(lang, line) if r.startswith("banned:")]


def test_a_longer_banned_match_is_reported_once():
    assert plain_reasons("es", "El pase fácil de Weso Wendez dentro del área se pierde.") == [
        "banned: line_es says 'dentro del área'"]


# --- honesty: a reason for the rating, never for the outcome ------------------------------------

def _cause(line: str) -> str:
    return f"outcome_cause: line_en says {line!r} away from a difficulty word: a reason may explain the rating, never the outcome"


EASY_REASON = dict(line_en="Weso Wendez's pass was easy because it was along the ground, but it goes astray.",
                   line_es="El pase de Weso Wendez era fácil porque iba raso, pero se pierde.",
                   line_el="Η πάσα του Weso Wendez ήταν εύκολη επειδή πήγε συρτά, αλλά χάνεται.")


def reason_out(**kw) -> NarratorOut:
    """A clean line for make_bundle() with driver 1 as the reason for the rating in every language."""
    return NarratorOut(**{**EASY_REASON, "drivers_used": [1], **kw})


def test_a_cause_of_the_outcome_is_rejected():
    out = reason_out(line_en="Weso Wendez's easy ball goes astray because it was along the ground.")
    assert reasons_for(out) == [_cause("because")]


def test_a_reason_for_the_rating_is_accepted():
    assert verify(reason_out(), make_bundle()).ok


@pytest.mark.parametrize("lang, line, marker", [
    ("en", "Weso Wendez's easy ball goes astray, thanks to the ground.", "thanks to"),
    ("en", "Weso Wendez's easy ball along the ground goes astray due to that.", "due to"),
    ("es", "El pase fácil y raso de Weso Wendez se pierde ya que nadie lo recibe.", "ya que"),
    ("es", "El pase fácil y raso de Weso Wendez se pierde, puesto que no llega.", "puesto que"),
    ("es", "El pase fácil y raso de Weso Wendez se pierde por culpa de la altura.", "por culpa de"),
    ("el", "Η εύκολη πάσα του Weso Wendez, συρτά, χάνεται αφού δεν βρίσκει κανέναν.", "αφού"),
    ("el", "Η εύκολη πάσα του Weso Wendez, συρτά, χάνεται διότι δεν βρίσκει κανέναν.", "διότι"),
    ("el", "Η εύκολη πάσα του Weso Wendez, συρτά, χάνεται χάρη σε μια κακή αναπήδηση.", "χάρη σε"),
    ("en", "Weso Wendez's easy ball goes astray, which makes it a waste.", "which makes it"),
    ("es", "El pase fácil y raso de Weso Wendez se pierde, lo que lo hace inútil.", "lo que lo hace"),
    ("el", "Η εύκολη πάσα του Weso Wendez χάνεται και ο Weso Wendez που την έκανε απογοητεύεται.", "που την έκανε"),
])
def test_causal_markers_away_from_a_difficulty_word(lang, line, marker):
    reasons = reasons_for(reason_out(**{f"line_{lang}": line}))
    assert [r for r in reasons if r.startswith("outcome_cause:")] == [
        f"outcome_cause: line_{lang} says {marker!r} away from a difficulty word: a reason may explain the rating, "
        f"never the outcome"]


@pytest.mark.parametrize("lang, line", [
    ("en", "What makes it easy is the ball along the ground, yet Weso Wendez's pass goes astray."),
    ("en", "Played along the ground, which made it easy, yet Weso Wendez's pass goes astray."),
    ("es", "El pase de Weso Wendez era fácil ya que iba raso, pero se pierde."),
    ("es", "Lo que hace fácil el pase raso de Weso Wendez no evita que se pierda."),
    ("es", "Iba raso, lo que lo hacía fácil, pero el pase de Weso Wendez se pierde."),
    ("el", "Η πάσα του Weso Wendez ήταν εύκολη διότι ήταν συρτά, αλλά χάνεται."),
    ("el", "Η πάσα του Weso Wendez ήταν εύκολη χάρη στο ότι ήταν συρτά, αλλά χάνεται."),
    ("el", "Πήγε συρτά, κάτι που την έκανε εύκολη, αλλά η πάσα του Weso Wendez χάνεται."),
])
def test_causal_markers_next_to_a_difficulty_word_are_accepted(lang, line):
    assert verify(reason_out(**{f"line_{lang}": line}), make_bundle()).ok


@pytest.mark.parametrize("lang, line", [
    ("en", "Weso Wendez's easy pass goes astray, even though it was along the ground."),
    ("es", "El pase fácil de Weso Wendez se pierde, aunque iba raso."),
    ("el", "Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που ήταν συρτά."),
])
def test_concessions_are_not_banned(lang, line):
    assert verify(line_out(**{f"line_{lang}": line}), make_bundle()).ok


def test_greek_delicate_and_especially_trip_nothing():
    out = line_out(line_el="Η λεπτή και ιδιαίτερα εύκολη πάσα του Weso Wendez χάνεται, παρόλο που πήγε συρτά.")
    result = verify(out, make_bundle())
    assert (result.ok, result.reasons) == (True, [])


# --- unsupported facts, the lead driver, a driver is needed --------------------------------------

def _unsupported(lang: str, said: str) -> str:
    return f"unsupported_fact: line_{lang} says {said!r}, which is not a driver in drivers_used"


def test_pressure_on_a_moment_with_no_pressure_driver_is_rejected():
    out = line_out(line_en="Weso Wendez's easy ball goes astray under pressure, even though it was along the ground.")
    assert reasons_for(out) == [_unsupported("en", "under pressure")]


def test_a_header_on_a_foot_pass_is_rejected():
    foot = [make_driver(1, *DRIVER_SPECS[0]), make_driver(2, "body_part", "body part", "Foot", "with the foot")]
    out = line_out(line_en="Weso Wendez's easy ball goes astray with a header, even though it was along the ground.")
    assert reasons_for(out, make_bundle(drivers=foot)) == [_unsupported("en", "with a header")]


def test_a_narrated_driver_left_out_of_drivers_used_is_rejected():
    lofted = [make_driver(1, "height", "pass height", "High Pass", "lofted high", "lowers")]
    out = NarratorOut(line_en="Weso Wendez's difficult lofted ball finds a teammate.",
                      line_es="El pase difícil de Weso Wendez encuentra a un compañero.",
                      line_el="Η δύσκολη πάσα του Weso Wendez βρίσκει συμπαίκτη.", drivers_used=[])
    assert reasons_for(out, hard_bundle(drivers=lofted)) == [
        _unsupported("en", "lofted"),
        "needs_driver: drivers_used is empty, but driver(s) [1] point the way of the rating (lowers) and the pass "
        "is rated hard (difficult)",
    ]


def test_a_longer_unsupported_phrase_is_reported_once():
    out = line_out(line_en="Weso Wendez's easy ball goes astray, even though it was along the ground and lofted high.")
    assert reasons_for(out) == [_unsupported("en", "lofted high")]   # not "lofted" again


def test_a_zone_noun_needs_its_position_driver():
    # seed 42, 81': "from the middle third" said, but the position driver was not in drivers_used
    out = line_out(line_en="Weso Wendez's easy pass goes astray, even though it was along the ground from the "
                           "middle third.",
                   line_es="El pase fácil de Weso Wendez se pierde, aunque iba raso desde el tercio medio.")
    assert reasons_for(out) == [_unsupported("en", "middle third"), _unsupported("es", "tercio medio")]


def test_the_phrase_scan_covers_the_whole_table_and_its_hints():
    for word in ("lofted", "ground", "low", "under pressure", "with a header", "out wide", "in the attacking third",
                 "from deep", "in midfield", "centrally", "middle third", "final third", "header"):
        assert word in TABLE_PHRASES["en"], word
    for word in ("συρτή", "ψηλή", "χαμηλή", "από την άμυνα", "από το κέντρο", "με κεφαλιά", "ύψους",
                 "αμυντικό τρίτο", "επιθετικό τρίτο"):
        assert word in TABLE_PHRASES["el"], word
    for word in ("de cabeza", "desde atrás", "en el centro del campo", "tercio medio", "último tercio"):
        assert word in TABLE_PHRASES["es"], word
    assert "with another part of the body" not in TABLE_PHRASES["en"]   # banned outright already


def test_mainly_needs_driver_one():
    out = NarratorOut(line_en="Weso Wendez's easy pass goes astray, even though it was mainly out wide.",
                      line_es="El pase fácil de Weso Wendez se pierde, aunque salió desde la banda.",
                      line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε από τα πλάγια.", drivers_used=[2])
    assert reasons_for(out) == [
        "lead_driver: line_en says 'mainly', but driver 1 is not a narrated driver in drivers_used"]
    lead = reason_out(line_en="Weso Wendez's pass was easy mainly because it was along the ground, but it goes astray.",
                      line_es="El pase de Weso Wendez era fácil sobre todo porque iba raso, pero se pierde.",
                      line_el="Η πάσα του Weso Wendez ήταν εύκολη κυρίως επειδή πήγε συρτά, αλλά χάνεται.")
    assert verify(lead, make_bundle()).ok


def test_an_empty_drivers_used_needs_no_matching_driver():
    assert reasons_for(bare_out()) == [
        "needs_driver: drivers_used is empty, but driver(s) [1, 2] point the way of the rating (raises) and the pass "
        "is rated easy (easy)"]


# --- every driver named in every line (hints mode) -----------------------------------------------

def test_a_driver_missing_from_one_language_is_rejected_in_hints_mode():
    out = make_out(line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που κύλησε στο χορτάρι.")
    assert reasons_for(out, mode="hints") == [
        "driver_expressed: line_el lists driver 1 (along the ground) but does not name it; every line names every "
        "driver in drivers_used"]
    assert verify(out, make_bundle(), "free").ok   # free mode shows the model only the English label


LOFTED = [make_driver(1, "height", "pass height", "High Pass", "lofted high", "lowers")]


def test_recognised_forms_name_a_driver():
    # the owner's S1 Greek form "λόγω του ύψους της"
    s1 = NarratorOut(line_en="Weso Wendez's pass was difficult because it was lofted high, but it finds a teammate.",
                     line_es="El pase de Weso Wendez era difícil porque iba por alto, pero encuentra a un compañero.",
                     line_el="Η πάσα του Weso Wendez ήταν δύσκολη λόγω του ύψους της, αλλά βρίσκει συμπαίκτη.",
                     drivers_used=[1])
    assert verify(s1, hard_bundle(drivers=LOFTED), "hints").ok
    # the zone nouns and the new hints
    middle = NarratorOut(line_en="Weso Wendez's easy pass goes astray, even though it came from the middle third.",
                         line_es="El pase fácil de Weso Wendez se pierde, aunque salió en el centro del campo.",
                         line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε από το κέντρο.",
                         drivers_used=[2])
    assert verify(middle, make_bundle(), "hints").ok


# --- a role for the drivers in every line, the same in each --------------------------------------

def test_a_driver_given_only_as_a_description_is_rejected():
    # seed 3001, 63': the Greek line puts "ψηλή" before πάσα and drops the reason the English line gives
    out = NarratorOut(line_en="Weso Wendez's pass was difficult because it was lofted high, but it finds a teammate.",
                      line_es="El pase de Weso Wendez era difícil porque iba por alto, pero encuentra a un compañero.",
                      line_el="Η ψηλή πάσα του Weso Wendez ήταν δύσκολη, αλλά βρίσκει συμπαίκτη.", drivers_used=[1])
    assert reasons_for(out, hard_bundle(drivers=LOFTED)) == [
        "role_parity: line_el names a driver but gives it no role: say it as the reason for the rating (next to the "
        "difficulty word) or as a concession (even though)"]


def test_the_lines_give_the_drivers_the_same_role():
    out = reason_out(line_es="El pase fácil de Weso Wendez se pierde, aunque iba raso.")
    assert reasons_for(out) == [
        "role_parity: the lines give the drivers different roles (line_en reason, line_es concession, line_el "
        "reason); all three take the same one"]


def test_a_line_with_no_driver_needs_no_role():
    assert verify(bare_out(), make_bundle(n_drivers=0)).ok


@pytest.mark.parametrize("lang, line, roles", [
    ("en", "Although it was along the ground, the pass goes astray.", {"concession"}),
    ("en", "Despite being lofted high, the pass finds a teammate.", {"concession"}),
    ("es", "Pese a ir por alto, el pase llega a un compañero.", {"concession"}),
    ("es", "A pesar de ir raso, el pase se pierde.", {"concession"}),
    ("el", "Αν και ήταν ψηλή, η πάσα βρίσκει συμπαίκτη.", {"concession"}),
    ("el", "Η πάσα χάνεται, παρ' όλο που ήταν συρτή.", {"concession"}),
    ("el", "Η πάσα χάνεται, ενώ ήταν συρτή.", {"concession"}),
    ("el", "Η πάσα ήταν δύσκολη λόγω του ύψους της, αλλά παρ' όλ' αυτά φτάνει σε συμπαίκτη.", {"reason"}),
    ("es", "El pase era difícil porque iba por alto, pero aun así llega a un compañero.", {"reason"}),
    ("el", "Η ψηλή πάσα ήταν δύσκολη, αλλά βρίσκει συμπαίκτη.", set()),
])
def test_line_roles(lang, line, roles):
    assert line_roles(lang, line) == roles


# --- one position piece per line (free and hints modes) -------------------------------------------

@pytest.mark.parametrize("mode", ["free", "hints"])
def test_both_position_pieces_are_rejected_outside_strict_mode(mode):
    out = NarratorOut(**dict(zip(("line_en", "line_es", "line_el"), POSITION_LINES["composed"])), drivers_used=[2])
    assert reasons_for(out, mode=mode) == [
        f"one_position_piece: line_{lang} names both the depth and the width of driver 2; use one of them"
        for lang in LANGS]
    for which in ("depth", "width"):
        piece = NarratorOut(**dict(zip(("line_en", "line_es", "line_el"), POSITION_LINES[which])), drivers_used=[2])
        assert verify(piece, make_bundle(), mode).ok


def test_centrally_counts_as_the_width_piece():
    # seed 3001, 88': "from their own defensive third centrally"
    deep = [make_driver(1, "position", "pitch position", "defensive_third|central",
                        "from their own defensive third, centrally")]
    out = NarratorOut(line_en="Weso Wendez's easy pass is lost, even though it was from their own defensive third "
                              "centrally.",
                      line_es="El pase fácil de Weso Wendez se pierde, aunque salió desde atrás.",
                      line_el="Η εύκολη πάσα του Weso Wendez χάνεται, παρόλο που έγινε από την άμυνα.", drivers_used=[1])
    assert reasons_for(out, make_bundle(drivers=deep), "hints") == [
        "one_position_piece: line_en names both the depth and the width of driver 1; use one of them"]


# --- rating_tense and double_contrast (the 74' line) ------------------------------------------------

from xplain.verifier import check_double_contrast, check_rating_tense  # noqa: E402

LINE_74 = NarratorOut(
    line_en="The pass from Garus Dornic is easy but does not reach a teammate, even though it was played along the ground.",
    line_es="Un pase de Garus Dornic es fácil pero no llega a un compañero, aunque iba raso.",
    line_el="Η μπαλιά του Garus Dornic ήταν εύκολη αλλά δεν φτάνει σε συμπαίκτη, παρόλο που ήταν συρτή.",
    drivers_used=[1])


def test_the_74_line_fails_both_new_checks():
    tense = check_rating_tense(LINE_74)
    assert tense == ["rating_tense: line_en rates the pass in the present ('is easy'); the rating is in the past",
                     "rating_tense: line_es rates the pass in the present ('es fácil'); the rating is in the past"]
    contrast = check_double_contrast(LINE_74)
    assert [r.split(":")[1].split()[0] for r in contrast] == ["line_en", "line_es", "line_el"]
    assert "'but' and 'even though'" in contrast[0]
    assert not verify(LINE_74, make_bundle(), "hints").ok


@pytest.mark.parametrize("lang, text", [
    ("en", "Weso Wendez's pass is easy, yet it goes astray."),
    ("en", "Weso Wendez's pass is very easy, yet it goes astray."),
    ("en", "These passes are easy, yet one goes astray."),
    ("es", "El pase de Weso Wendez es muy fácil, pero se pierde."),
    ("es", "El pase de Weso Wendez es complicado, pero llega."),
    ("el", "Η πάσα του Weso Wendez είναι εύκολη, αλλά χάνεται."),
    ("el", "Η πάσα του Weso Wendez είναι πολύ δύσκολη, αλλά φτάνει."),
])
def test_present_tense_ratings_are_rejected(lang, text):
    out = make_out(**{f"line_{lang}": text})
    [reason] = check_rating_tense(out)
    assert reason.startswith(f"rating_tense: line_{lang} rates the pass in the present")


@pytest.mark.parametrize("lang, text", [
    ("en", "Weso Wendez's easy pass goes astray, even though it was along the ground."),   # possessive 's
    ("en", "Weso Wendez's pass was easy, yet it is lost."),                                 # "is lost": an outcome
    ("en", "It is Weso Wendez whose easy pass goes astray."),
    ("es", "El pase de Weso Wendez era fácil, pero se pierde."),
    ("el", "Η πάσα του Weso Wendez ήταν εύκολη, αλλά χάνεται."),
])
def test_past_ratings_and_other_present_verbs_pass(lang, text):
    assert check_rating_tense(make_out(**{f"line_{lang}": text})) == []


def test_one_contrast_per_line():
    assert check_double_contrast(make_out()) == []   # a concession only
    but_only = make_out(line_en="Weso Wendez's pass was easy because it was along the ground, but it goes astray.")
    assert check_double_contrast(but_only) == []
    both = make_out(line_el="Ο Weso Wendez πασάρει εύκολα, αλλά η πάσα του χάνεται, παρόλο που πήγε συρτά.")
    assert check_double_contrast(both) == ["double_contrast: line_el has both 'αλλά' and 'παρόλο που'; one "
                                           "contrast per line"]


def test_the_worked_examples_pass_the_new_checks():
    from xplain.prompts import EXAMPLES
    for ex in EXAMPLES:
        out = NarratorOut(line_en=ex.line_en, line_es=ex.line_es, line_el=ex.line_el,
                          drivers_used=list(ex.drivers_used))
        assert check_rating_tense(out) == [] and check_double_contrast(out) == [], ex.name
