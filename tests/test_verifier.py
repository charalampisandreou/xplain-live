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
from xplain.verifier import EASY_WORDS, HARD_WORDS, LANGS, MAX_WORDS, difficulty_words, verify

LABELS = ("along the ground", "from the middle third, out wide", "with no pressure on the passer")


def make_bundle(xp: float = 0.94, minute: int = 3, n_drivers: int = 3, completed: bool = False,
                added: int = 0) -> MomentBundle:
    """A small Moment + Explanation built by hand, with Analyst-style evidence."""
    event_id = "m1-00042"
    d = analytics.difficulty(xp)
    dlt = analytics.delta(completed, xp)
    evidence = build_evidence(event_id, {"xp": xp, "delta": dlt, "difficulty": d, "minute": minute,
                                         "minute_added": added, "minute_label": clock_label(minute, added),
                                         "completed": completed, "player_name": "Weso Wendez"})
    moment = Moment(
        moment_id=f"moment:{event_id}", kind="surprise_success" if completed else "surprise_failure",
        event_id=event_id, match_id="m1", seq=42, t=(minute - 1) * 60.0 + 5, minute=minute, minute_added=added,
        minute_label=clock_label(minute, added), period=1,
        team_id="home", player_id="p7", player_name="Weso Wendez", completed=completed, xp=xp, delta=dlt,
        difficulty=d, band=analytics.difficulty_band(d), significance=abs(dlt),
        pass_geometry=PassGeometry(start_x=50, start_y=75, end_x=70, end_y=70, length=20.6, ball_speed_ms=14),
        evidence=evidence, model_version="test",
    )
    drivers = [Driver(rank=i + 1, feature="position", topic="pitch position", value="x", label=LABELS[i],
                      shap=0.5, direction="raises") for i in range(n_drivers)]
    explanation = Explanation(
        moment_id=moment.moment_id, event_id=event_id, method="test", unit="log-odds", baseline=0.0,
        raw_score=0.5 * n_drivers, drivers=drivers, rest=0.0, evidence={}, model_version="test",
    )
    return MomentBundle(moment=moment, explanation=explanation, qualifies_live=False, qualified_by=())


def hard_bundle(**kw) -> MomentBundle:
    """A completed pass the model rated hard (surprise_success, xP 0.25: tier "difficult")."""
    return make_bundle(xp=0.25, completed=True, **kw)


def make_out(**kw) -> NarratorOut:
    """A clean line for make_bundle(): an easy pass along the ground that goes astray."""
    base = dict(
        line_en="Weso Wendez's easy ball along the ground goes astray.",
        line_es="El pase fácil a ras de suelo de Weso Wendez se pierde.",
        line_el="Η εύκολη πάσα του Weso Wendez κατά μήκος του εδάφους χάνεται.",
        drivers_used=[1],
    )
    return NarratorOut(**{**base, **kw})


HARD_OUT = dict(
    line_en="Weso Wendez gets a difficult ball away and finds a teammate.",
    line_es="Weso Wendez saca un pase difícil y encuentra a un compañero.",
    line_el="Ο Weso Wendez βγάζει μια δύσκολη πάσα και βρίσκει συμπαίκτη.",
    drivers_used=[1],
)


def reasons_for(out: NarratorOut, bundle: MomentBundle | None = None) -> list[str]:
    return verify(out, bundle or make_bundle()).reasons


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
    out = NarratorOut(
        line_en=f"{name}'s {tier.en} pass goes astray.",
        line_es=f"El pase {tier.es} de {name} se pierde.",
        line_el=f"Η {tier.el} πάσα του {name} χάνεται.",
        drivers_used=[d.rank for d in bundle.explanation.drivers[:1]],
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
    ("en", "Wendez's tough ball along the ground goes astray."),
    ("es", "El pase difícil de Wendez se pierde."),
    ("el", "Η δύσκολη πάσα του Wendez χάνεται."),
])
def test_hard_word_on_an_easy_pass_is_rejected(lang, line):
    reasons = reasons_for(make_out(**{f"line_{lang}": line}))
    assert len(reasons) == 1
    assert reasons[0].startswith(f"difficulty: line_{lang} says ")
    assert reasons[0].endswith("but the pass is rated easy (easy)")


@pytest.mark.parametrize("lang, line", [
    ("en", "Weso Wendez's simple ball finds a teammate."),
    ("es", "El pase sencillo de Weso Wendez encuentra a un compañero."),
    ("el", "Η εύκολη πάσα του Weso Wendez βρίσκει συμπαίκτη."),
])
def test_easy_word_on_a_hard_pass_is_rejected(lang, line):
    out = NarratorOut(**{**HARD_OUT, f"line_{lang}": line})
    reasons = reasons_for(out, hard_bundle())
    assert len(reasons) == 1
    assert reasons[0].startswith(f"difficulty: line_{lang} says ")
    assert reasons[0].endswith("but the pass is rated hard (difficult)")


def test_greek_accent_variants_are_found():
    assert difficulty_words("el", "μια ευκολότερη πάσα")["easy"]
    assert difficulty_words("el", "ΔΥΣΚΟΛΗ ΠΑΣΑ")["hard"]


def test_line_without_a_difficulty_word_is_accepted_but_noted():
    out = make_out(line_en="Weso Wendez's ball along the ground goes astray.",
                   line_el="Η πάσα του Weso Wendez κατά μήκος του εδάφους χάνεται.")
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
    assert reason in reasons_for(make_out(**{f"line_{lang}": line}))


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
    assert reasons_for(make_out(**{f"line_{lang}": line})) == [f"no_numbers: line_{lang} says {word!r}"]


def test_greek_delicate_pass_is_not_a_minute():
    out = make_out(line_el="Η λεπτή και εύκολη πάσα του Weso Wendez χάνεται.")
    assert verify(out, make_bundle()).ok


def test_spike_lines_with_numbers_are_now_rejected():
    out = NarratorOut(
        line_en="At 3 minutes, Weso Wendez makes a 94% pass attempt from the middle third, out wide, but fails.",
        line_es="A los 3 minutos, Weso Wendez intenta un pase del 94% desde el tercio medio, hacia afuera, pero falla.",
        line_el="Στο 3ο λεπτό, ο Weso Wendez επιχειρεί μια πάσα 94% από το μεσαίο τρίτο, προς τα έξω, αλλά αποτυγχάνει.",
        drivers_used=[2],
    )
    reasons = reasons_for(out)
    assert {r.split(":")[0] for r in reasons} == {"no_numbers"}
    assert {r.split(" ")[1] for r in reasons} == {"line_en", "line_es", "line_el"}


# --- length and sentence ------------------------------------------------------------------------

def test_word_limits():
    assert MAX_WORDS == {"en": 18, "es": 22, "el": 22}


@pytest.mark.parametrize("lang", LANGS)
def test_over_length_line_is_rejected(lang):
    n = MAX_WORDS[lang] + 1
    line = " ".join(["word"] * n) + "."
    assert reasons_for(make_out(**{f"line_{lang}": line})) == [f"length: line_{lang} has {n} words, max {MAX_WORDS[lang]}"]


@pytest.mark.parametrize("lang", LANGS)
def test_exactly_max_words_passes(lang):
    line = " ".join(["word"] * MAX_WORDS[lang]) + "."
    assert verify(make_out(**{f"line_{lang}": line}), make_bundle()).ok


@pytest.mark.parametrize("line, reason", [
    ("", "sentence: line_en is empty"),
    ("   ", "sentence: line_en is empty"),
    ("An easy ball along the ground\ngoes astray.", "sentence: line_en has a line break"),
    ("An easy ball along the ground. It goes astray.", "sentence: line_en has more than one sentence"),
])
def test_line_must_be_one_non_empty_sentence(line, reason):
    assert reasons_for(make_out(line_en=line)) == [reason]


def test_greek_question_mark_ends_a_sentence():
    out = make_out(line_el="Χάνεται η εύκολη πάσα; Ναι, του Weso Wendez.")
    assert reasons_for(out) == ["sentence: line_el has more than one sentence"]


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
    reasons = reasons_for(make_out(**{field: line}))
    assert any(r.startswith(f"banned: {field} says ") for r in reasons), reasons


def test_throw_in_is_not_a_throw():
    out = make_out(line_en="Wendez misses an easy pass from a throw-in.")
    assert verify(out, make_bundle()).ok


def test_spanish_lanza_is_allowed():
    out = make_out(line_es="Wendez lanza un pase fácil y lo pierde.")
    assert verify(out, make_bundle()).ok


def test_banned_is_case_insensitive():
    reasons = reasons_for(make_out(line_en="Wendez misses an easy pass INSIDE THE BOX."))
    assert reasons == ["banned: line_en says 'INSIDE THE BOX'"]


# --- drivers ------------------------------------------------------------------------------------

def test_driver_rank_that_does_not_exist():
    reasons = reasons_for(make_out(drivers_used=[1, 4]), make_bundle(n_drivers=3))
    assert reasons == ["drivers: drivers_used has [4], the moment's drivers are [1, 2, 3]"]


def test_duplicate_driver_rank():
    assert reasons_for(make_out(drivers_used=[2, 2])) == ["drivers: drivers_used repeats [2]"]


def test_zero_drivers_with_empty_drivers_used_passes():
    assert verify(make_out(drivers_used=[]), make_bundle(n_drivers=0)).ok


def test_zero_drivers_with_drivers_used_fails():
    reasons = reasons_for(make_out(drivers_used=[1]), make_bundle(n_drivers=0))
    assert reasons == ["drivers: the moment has no drivers but drivers_used is [1]"]


# --- several failures ---------------------------------------------------------------------------

def test_two_failures_return_two_reasons():
    out = make_out(line_en="Wendez misses an easy pass inside the box.", drivers_used=[5])
    result = verify(out, make_bundle())
    assert not result.ok
    assert result.reasons == [
        "banned: line_en says 'inside the box'",
        "drivers: drivers_used has [5], the moment's drivers are [1, 2, 3]",
    ]


def test_word_lists_compile_for_every_language():
    for table in (EASY_WORDS, HARD_WORDS):
        assert set(table) == set(LANGS)
        for patterns in table.values():
            for p in patterns:
                re.compile(p)
