import pytest
from pydantic import ValidationError

from xplain import analytics
from xplain.analyst import build_evidence
from xplain.clock import clock_label
from xplain.contracts import Driver, Explanation, Moment, MomentBundle, PassGeometry
from xplain.generator import generate_match
from xplain.narration import NarratorOut
from xplain.pipeline import process_match
from xplain.scoring import ModelLoadError, XPassScorer, find_default_model
from xplain.verifier import MAX_WORDS, allowed_numbers, verify

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


def make_out(**kw) -> NarratorOut:
    base = dict(
        line_en="Weso Wendez misses a 94% pass in minute 3, played along the ground.",
        line_es="Weso Wendez falla un pase del 94% en el minuto 3, a ras de suelo.",
        line_el="Στο 3ο λεπτό ο Weso Wendez χάνει μια πάσα 94% κατά μήκος του εδάφους.",
        numbers_used=["94", "3"],
        drivers_used=[1],
    )
    return NarratorOut(**{**base, **kw})


def reasons_for(out: NarratorOut, bundle: MomentBundle | None = None) -> list[str]:
    return verify(out, bundle or make_bundle()).reasons


# --- contract -----------------------------------------------------------------------------------

def test_narrator_out_is_frozen_and_strict():
    out = make_out()
    with pytest.raises(ValidationError):
        out.line_en = "x"
    with pytest.raises(ValidationError):
        NarratorOut(**out.model_dump(), extra="x")


def test_percent_is_round_100x():
    assert analytics.percent(0.94) == 94
    assert analytics.percent(0.125) == 12      # round half to even
    assert analytics.percent(0.145) == 14      # 100 * 0.145 == 14.499999999999998
    assert analytics.difficulty(0.94) == 6


# --- passing ------------------------------------------------------------------------------------

def test_clean_line_passes():
    result = verify(make_out(), make_bundle())
    assert result.ok and result.reasons == []


def test_allowed_numbers_are_xp_percent_minute_and_difficulty():
    assert allowed_numbers(make_bundle(xp=0.94, minute=3)) == {"94", "3", "6"}


def test_greek_ordinal_counts_as_its_digits():
    out = make_out(line_el="Στο 3ο λεπτό ο Weso Wendez χάνει μια πάσα 94%.")
    assert verify(out, make_bundle()).ok


def test_real_seed42_bundle_passes():
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc})")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path}")
    bundle = process_match(generate_match(42)[0], XPassScorer(model_path, feature_list_path))[0]
    m = bundle.moment
    pct, minute = str(analytics.percent(m.xp)), str(m.minute)
    out = NarratorOut(
        line_en=f"In minute {minute}, a {pct}% pass goes astray.",
        line_es=f"En el minuto {minute}, un pase del {pct}% se pierde.",
        line_el=f"Στο {minute}ο λεπτό χάνεται μια πάσα {pct}%.",
        numbers_used=[pct, minute],
        drivers_used=[d.rank for d in bundle.explanation.drivers[:1]],
    )
    assert verify(out, bundle).reasons == []


# --- numbers ------------------------------------------------------------------------------------

def test_wrong_percent_is_rejected():
    out = make_out(line_en="Weso Wendez misses a 95% pass in minute 3.", numbers_used=["95", "94", "3"])
    assert reasons_for(out) == ["numbers: numbers_used has '95', not in the evidence (allowed ['3', '6', '94'])"]


def test_rounding_edge_uses_round_100x():
    bundle = make_bundle(xp=0.145, completed=True)      # 100 * 0.145 rounds to 14, not 15
    good = make_out(line_en="A 14% pass finds its man in minute 3.", line_es="Un pase del 14% llega en el minuto 3.",
                    line_el="Μια πάσα 14% βρίσκει αποδέκτη στο 3ο λεπτό.", numbers_used=["14", "3"])
    bad = good.model_copy(update={"line_en": "A 15% pass finds its man in minute 3.", "numbers_used": ["15", "14", "3"]})
    assert verify(good, bundle).ok
    assert reasons_for(bad, bundle) == ["numbers: numbers_used has '15', not in the evidence (allowed ['3', '14', '86'])"]


def test_number_in_line_missing_from_numbers_used():
    assert reasons_for(make_out(numbers_used=["94"])) == [
        "numbers: line_en has 3, not in numbers_used",
        "numbers: line_es has 3, not in numbers_used",
        "numbers: line_el has 3, not in numbers_used",
    ]


def test_numbers_used_entry_matching_nothing_in_evidence():
    assert reasons_for(make_out(numbers_used=["94", "3", "40"])) == [
        "numbers: numbers_used has '40', not in the evidence (allowed ['3', '6', '94'])",
    ]


def test_numbers_used_must_be_digits_as_written():
    reasons = reasons_for(make_out(numbers_used=["94%", "3"]))
    assert "numbers: numbers_used has '94%', not in the evidence (allowed ['3', '6', '94'])" in reasons


# --- length and sentence ------------------------------------------------------------------------

def test_word_limits():
    assert MAX_WORDS == {"en": 18, "es": 22, "el": 22}


@pytest.mark.parametrize("lang", ["en", "es", "el"])
def test_over_length_line_is_rejected(lang):
    n = MAX_WORDS[lang] + 1
    line = " ".join(["word"] * n) + "."
    assert reasons_for(make_out(**{f"line_{lang}": line})) == [f"length: line_{lang} has {n} words, max {MAX_WORDS[lang]}"]


@pytest.mark.parametrize("lang", ["en", "es", "el"])
def test_exactly_max_words_passes(lang):
    line = " ".join(["word"] * MAX_WORDS[lang]) + "."
    assert verify(make_out(**{f"line_{lang}": line}), make_bundle()).ok


SPIKE_ES = "A los 3 minutos, Weso Wendez intenta un pase del 94% desde el tercio medio, hacia afuera, pero falla."


@pytest.mark.parametrize("line_en, line_el", [
    # direct call: en 18, es 19, el 19 words
    ("At 3 minutes, Weso Wendez makes a 94% pass attempt from the middle third, out wide, but fails.",
     "Στο 3ο λεπτό, ο Weso Wendez επιχειρεί μια πάσα 94% από το μεσαίο τρίτο, προς τα έξω, αλλά αποτυγχάνει."),
    # workflow call: en 17, es 19, el 18 words
    ("At 3 minutes, Weso Wendez attempts a 94% pass from the middle third, out wide, but fails.",
     "Στο 3ο λεπτό, η πάσα 94% του Weso Wendez από το μεσαίο τρίτο, προς τα έξω, αποτυγχάνει απροσδόκητα."),
])
def test_spike_samples_pass(line_en, line_el):
    """The two gpt-4.1-mini outputs from the spike (moment m42-00042: 94% pass, minute 3) pass at the new limits."""
    out = NarratorOut(line_en=line_en, line_es=SPIKE_ES, line_el=line_el, numbers_used=["3", "94"], drivers_used=[2])
    assert verify(out, make_bundle(xp=0.94, minute=3, n_drivers=3)).reasons == []


@pytest.mark.parametrize("line, reason", [
    ("", "sentence: line_en is empty"),
    ("   ", "sentence: line_en is empty"),
    ("A 94% pass in minute 3\ngoes astray.", "sentence: line_en has a line break"),
    ("A 94% pass in minute 3. It goes astray.", "sentence: line_en has more than one sentence"),
])
def test_line_must_be_one_non_empty_sentence(line, reason):
    assert reasons_for(make_out(line_en=line)) == [reason]


def test_greek_question_mark_ends_a_sentence():
    out = make_out(line_el="Χάνεται η πάσα 94%; Ναι, στο 3ο λεπτό.")
    assert reasons_for(out) == ["sentence: line_el has more than one sentence"]


# --- banned claims ------------------------------------------------------------------------------

@pytest.mark.parametrize("field, line", [
    # box / penalty area
    ("line_en", "Wendez misses a 94% pass inside the box."),
    ("line_es", "Wendez falla un pase del 94% dentro del área."),
    ("line_el", "Ο Wendez χάνει πάσα 94% μέσα στην περιοχή."),
    # line-breaking
    ("line_en", "Wendez tries a line-breaking 94% pass."),
    ("line_en", "Wendez's 94% pass breaks the line."),
    ("line_es", "El pase del 94% de Wendez rompe las líneas."),
    ("line_el", "Η πάσα 94% του Wendez σπάει τις γραμμές."),
    # defender counts
    ("line_en", "Wendez's 94% pass beats three defenders."),
    ("line_es", "El pase del 94% de Wendez supera a dos rivales."),
    ("line_el", "Η πάσα 94% του Wendez περνά δύο αμυντικούς."),
    # left / right
    ("line_en", "Wendez misses a 94% pass on the left."),
    ("line_en", "Wendez misses a 94% pass with his right foot."),
    ("line_es", "Wendez falla un pase del 94% por la izquierda."),
    ("line_el", "Ο Wendez χάνει πάσα 94% από τα δεξιά."),
    # model internals
    ("line_en", "SHAP says the 94% pass fails."),
    ("line_en", "The 94% pass fails against the baseline."),
    ("line_en", "The raw score says 94% and the pass fails."),
    ("line_en", "The expected pass value is 94%."),
    ("line_es", "La línea base dice 94%."),
    ("line_el", "Η αναμενόμενη αξία είναι 94%."),
    # throw
    ("line_en", "Wendez throws away a 94% pass."),
    ("line_el", "Ο Wendez ρίχνει μια πάσα 94% και χάνει."),
])
def test_banned_claims(field, line):
    reasons = reasons_for(make_out(**{field: line}))
    assert any(r.startswith(f"banned: {field} says ") for r in reasons), reasons


def test_throw_in_is_not_a_throw():
    out = make_out(line_en="Wendez misses a 94% pass from a throw-in.")
    assert verify(out, make_bundle()).ok


def test_spanish_lanza_is_allowed():
    out = make_out(line_es="Wendez lanza un pase del 94% en el minuto 3 y falla.")
    assert verify(out, make_bundle()).ok


def test_banned_is_case_insensitive():
    reasons = reasons_for(make_out(line_en="Wendez misses a 94% pass INSIDE THE BOX."))
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
    out = make_out(line_en="Wendez misses a 94% pass in minute 3 inside the box.", drivers_used=[5])
    result = verify(out, make_bundle())
    assert not result.ok
    assert result.reasons == [
        "banned: line_en says 'inside the box'",
        "drivers: drivers_used has [5], the moment's drivers are [1, 2, 3]",
    ]
