"""Recap v2: the star player, the slots, and the recap_slots checks on the raw text."""
from xplain.prompts import RECAP_EXAMPLES
from xplain.recap import (STAR_MIN_PVA, half_up, recap_claims, recap_facts, slot_directions, slot_values,
                          star_player, team_passing)
from xplain.recap_verifier import RECAP_WORD_LIMIT, verify_recap

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_recap import card, marks  # noqa: F401  (fixture)

_, HT, DRAW_FT = RECAP_EXAMPLES
from .recap_fixtures import FT  # noqa: E402  (frozen case)
PLAYERS = ["Dario Pell", "Joss Marlin", "Tomas Avery", "Ollie Branck", "Ivo Grant"]


def check(example, **update):
    return verify_recap(example.out.model_copy(update=update), example.card, PLAYERS)


def sub(example, lang, old, new):
    text = getattr(example.out, f"text_{lang}")
    assert old in text, old
    return {f"text_{lang}": text.replace(old, new)}


def failed(verdict, name):
    return [r for r in verdict.reasons if r.startswith(name)]


# --- star player ---------------------------------------------------------------------------------------

def fake_marks(marks, spec):
    """spec: (player, team, period, [(completed, xp), ...]); copies of a real PassMark, seq in order."""
    base, out, seq = marks[0], [], 0
    for player, team, period, passes in spec:
        for completed, xp in passes:
            seq += 1
            out.append(base.model_copy(update={"player_id": player, "player_name": player.title(), "team_id": team,
                                               "team_name": team.upper(), "period": period, "seq": seq,
                                               "completed": completed, "xp": xp}))
    return out


def test_half_up_rounds_halves_away_from_zero():
    assert (half_up(43.5), half_up(44.5), half_up(393.6), half_up(336.1), half_up(2.4999)) == (44, 45, 394, 336, 2)


def test_the_star_is_the_top_pva_with_ties_to_more_passes(marks):  # noqa: F811
    spec = [("ann", "h", 1, [(True, 0.0)] * 3),                     # pva 3, 3 passes
            ("bob", "a", 1, [(True, 0.0)] * 3 + [(False, 0.0)]),    # pva 3, 4 passes: wins the tie
            ("cyd", "h", 2, [(True, 0.0)] * 5)]                     # pva 5, second half only
    ft = star_player(fake_marks(marks, spec), "full_time")
    assert (ft.name, ft.pva, ft.passes, ft.completed, ft.expected) == ("Cyd", 5, 5, 5, 0)
    ht = star_player(fake_marks(marks, spec), "half_time")
    assert (ht.name, ht.pva, ht.passes, ht.team_id) == ("Bob", 3, 4, "a")


def test_the_star_and_the_bands_are_gated_on_the_shown_two_decimal_value(marks):  # noqa: F811
    # the star: 3 completed passes; xP 1.49 -> shown +1.51 (the whole number would be 2): no star
    assert star_player(fake_marks(marks, [("ann", "h", 1, [(True, 0.49), (True, 0.5), (True, 0.5)])]),
                       "full_time") is None
    s = star_player(fake_marks(marks, [("ann", "h", 1, [(True, 0.5), (True, 0.25), (True, 0.25)])]), "full_time")
    assert (s.vs_expected_xp, s.pva) == (2.0, 2)   # exactly +2.00: a star
    # the bands (full time: 8): +7.51 is close (the whole number would be 8), +8.00 above, -8.00 below
    near = fake_marks(marks, [("ann", "h", 1, [(True, 0.249)] * 10)])
    t = team_passing(near, "h", "H", "full_time")
    assert (t.vs_expected_xp, t.vs_expected, t.band) == (7.51, 8, "close")
    above = team_passing(fake_marks(marks, [("ann", "h", 1, [(True, 0.25)] * 8 + [(True, 0.0)] * 2)]),
                         "h", "H", "full_time")
    assert (above.vs_expected_xp, above.band) == (8.0, "above")
    below = team_passing(fake_marks(marks, [("ann", "h", 1, [(False, 0.5)] * 16)]), "h", "H", "full_time")
    assert (below.vs_expected_xp, below.vs_expected, below.band) == (-8.0, -8, "below")


def test_no_star_below_the_minimum(marks):  # noqa: F811
    assert STAR_MIN_PVA == 2
    one = fake_marks(marks, [("ann", "h", 1, [(True, 0.3), (True, 0.3)])])                # 2 - half_up(0.6) = 1
    assert star_player(one, "full_time") is None
    two = fake_marks(marks, [("ann", "h", 1, [(True, 0.2), (True, 0.2), (True, 0.2)])])   # 3 - half_up(0.6) = 2
    assert star_player(two, "full_time").pva == 2


def test_the_star_on_the_card_and_its_claim(seed42_events, marks):  # noqa: F811
    c = card(seed42_events, marks)
    assert c.star == star_player(marks, "full_time")
    if c.star is not None:
        assert recap_claims(c)[f"star:{c.star.name}"].endswith("in the match")
        assert recap_facts(c)["star"]["team"] == c.star.team_name


def test_slots_carry_two_decimals_with_each_languages_decimal_mark():
    from xplain.recap import slot_text
    from xplain.recap_verifier import CLAUSE_SPLIT, fill, verify_recap
    from .recap_fixtures import FT as FIXTURE
    c = FIXTURE.card
    card2 = c.model_copy(update={
        "home": c.home.model_copy(update={"expected_xp": 319.55, "vs_expected_xp": 15.45}),
        "away": c.away.model_copy(update={"expected_xp": 334.89, "vs_expected_xp": -14.89}),
        "star": c.star.model_copy(update={"expected_xp": 39.52, "vs_expected_xp": 6.48})})
    assert slot_values(card2) == {"home_vs_expected": "15.45", "away_vs_expected": "14.89",
                                  "star_name": "Ollie Branck", "star_pva": "6.48"}
    assert slot_text("15.45", "en") == "15.45" and slot_text("15.45", "es") == slot_text("15.45", "el") == "15,45"
    assert slot_text("Ollie Branck", "es") == "Ollie Branck"
    filled = fill(FIXTURE.out, card2)
    assert "completed 15.45 more passes" in filled.text_en and "completó 15,45 pases más" in filled.text_es
    assert "ολοκλήρωσε 15,45 περισσότερες" in filled.text_el and "με 6,48 περισσότερες" in filled.text_el
    # a decimal mark between digits is not a clause break; the text passes every check
    assert len(CLAUSE_SPLIT.split("completó 15,45 pases; 2.5 más, y")) == 3
    assert verify_recap(FIXTURE.out, card2, ["Dario Pell", "Joss Marlin", "Tomas Avery", "Ollie Branck"]).ok


def test_slots_exist_only_for_above_or_below_teams_and_a_star():
    assert slot_values(FT.card) == {"home_vs_expected": "15", "away_vs_expected": "15", "star_name": "Ollie Branck",
                                    "star_pva": "6"}
    assert slot_directions(FT.card) == {"home_vs_expected": "more", "away_vs_expected": "fewer", "star_pva": "more"}
    assert slot_values(HT.card) == {"away_vs_expected": "15", "star_name": "Dario Pell", "star_pva": "4"}
    assert slot_values(DRAW_FT.card) == {"home_vs_expected": "15"}   # away is close, no star
    assert set(recap_facts(DRAW_FT.card)["slots"]) == {"home_vs_expected"}


# --- recap_slots and the filled text ---------------------------------------------------------------------

def test_the_live_ft1_example_has_the_late_winner_scorer_as_star():
    ft1 = RECAP_EXAMPLES[0]
    v = verify_recap(ft1.out, ft1.card, PLAYERS)
    assert v.ok, v.reasons
    assert v.filled.text_en.endswith("Tomas Avery also stood out with 6 more passes than expected.")
    assert v.filled.text_en.count("late winner") == 1   # the goal is named once, in the result sentence
    assert "también destacó con 6" in v.filled.text_es and "ξεχώρισε επίσης με 6" in v.filled.text_el
    assert recap_facts(ft1.card)["star"] == {"team": "Harbourne Athletic", "scored": True, "scored_late_winner": True,
                                             "scored_late_equaliser": False}
    assert recap_claims(ft1.card)["late_winner:Harbourne Athletic"].endswith("through Tomas Avery")


def test_the_star_who_scored_with_commas_is_rejected_by_the_clause_binding():
    ft1 = RECAP_EXAMPLES[0]
    comma = ft1.out.model_copy(update={"text_en": ft1.out.text_en.replace(
        "{star_name} also stood out", "{star_name}, who scored the late winner, stood out")})
    v = verify_recap(comma, ft1.card, PLAYERS)
    assert any("{star_pva} in a clause without {star_name}" in r for r in failed(v, "recap_slots"))


def test_star_facts_when_the_star_did_not_score():
    assert recap_facts(FT.card)["star"] == {"team": "Harbourne Athletic", "scored": False,
                                            "scored_late_winner": False, "scored_late_equaliser": False}


def test_the_three_examples_pass_and_fill():
    for ex in RECAP_EXAMPLES:
        v = verify_recap(ex.out, ex.card, PLAYERS)
        assert v.ok, (ex.name, v.reasons)
        for lang in ("en", "es", "el"):
            filled = getattr(v.filled, f"text_{lang}")
            assert "{" not in filled and len(filled.split()) <= RECAP_WORD_LIMIT
    filled = verify_recap(FT.out, FT.card, PLAYERS).filled
    assert filled.text_en.endswith("Ollie Branck stood out with 6 more passes than expected.")
    assert "Kellsworth Rovers completed 15 fewer than expected" in filled.text_en


def test_a_below_team_slot_written_with_more_is_rejected():
    v = check(FT, **sub(FT, "en", "{away_vs_expected} fewer than expected", "{away_vs_expected} more than expected"))
    assert any("{away_vs_expected} with the 'more' phrase; it must be 'fewer" in r for r in failed(v, "recap_slots"))


def test_a_slot_without_its_expected_phrase_is_rejected():
    v = check(FT, **sub(FT, "es", "{home_vs_expected} pases más de lo esperado", "{home_vs_expected} pases buenos"))
    assert any("text_es does not follow {home_vs_expected} with the 'more ... than expected' phrase" in r
               for r in failed(v, "recap_slots"))
    v = check(FT, **sub(FT, "el", "{star_pva} περισσότερες", "{star_pva} καλές"))
    assert any("text_el does not follow {star_pva}" in r for r in failed(v, "recap_slots"))


def test_the_star_slot_takes_more_only():
    v = check(FT, **sub(FT, "en", "{star_pva} more passes than expected", "{star_pva} fewer passes than expected"))
    assert any("{star_pva} with the 'fewer' phrase" in r for r in failed(v, "recap_slots"))


def test_unknown_unavailable_and_mismatched_slots():
    v = check(FT, **sub(FT, "en", "{star_pva}", "{star_passes}"))
    assert any("unknown slot {star_passes}" in r for r in failed(v, "recap_slots"))
    assert any("differ across languages" in r for r in failed(v, "recap_slots"))
    v = check(DRAW_FT, **sub(DRAW_FT, "en", "Kellsworth Rovers passed about as expected",
                             "Kellsworth Rovers completed {away_vs_expected} fewer than expected"))
    assert any("{away_vs_expected}, which this card does not have" in r for r in failed(v, "recap_slots"))
    assert any("unfilled slot" in r for r in failed(v, "recap_slots"))


def test_slots_need_their_clause_and_claim():
    v = check(FT, **sub(FT, "en", "Kellsworth Rovers completed {away_vs_expected}", "they completed {away_vs_expected}"))
    assert any("does not name Kellsworth Rovers" in r for r in failed(v, "recap_slots"))
    v = check(FT, claims_used=[c for c in FT.out.claims_used if c != "star:Ollie Branck"])
    assert any("without the claim 'star:Ollie Branck'" in r for r in failed(v, "recap_slots"))
    v = check(FT, **sub(FT, "en", "{star_name} stood out with {star_pva}", "{star_name} stood out, with {star_pva}"))
    assert any("{star_pva} in a clause without {star_name}" in r for r in failed(v, "recap_slots"))
    v = check(FT, claims_used=[c for c in FT.out.claims_used if c != "passing:Kellsworth Rovers"])
    assert any("without the claim 'passing:Kellsworth Rovers'" in r for r in failed(v, "recap_slots"))


def test_braces_digits_and_number_words_in_the_raw_text():
    v = check(FT, **sub(FT, "en", "{star_name} stood", "{star_name stood"))
    assert any("brace" in r for r in failed(v, "recap_slots"))
    v = check(FT, **sub(FT, "en", "{home_vs_expected} more passes", "15 more passes"))
    assert failed(v, "no_numbers") and any("differ across languages" in r for r in v.reasons)
    v = check(FT, **sub(FT, "es", "{star_pva} pases", "dos pases"))
    assert failed(v, "no_numbers")


def test_length_and_sentences_count_the_filled_text():
    longer = FT.out.text_en.replace("stood out with", "stood out once again on the day with")   # 48 -> 52 words
    assert check(FT, text_en=longer).ok
    v = check(FT, text_en=longer + " Harbourne Athletic and Kellsworth Rovers then left the field together.")
    assert failed(v, "format")   # a fourth sentence


def test_a_star_clause_must_say_more():
    v = check(FT, **sub(FT, "en", "{star_name} stood out with {star_pva} more passes than expected.",
                        "{star_name} stood out with {star_pva} more passes than expected, though Ollie Branck "
                        "passed below expected."))
    assert any("the star Ollie Branck passed 'below' expected" in r for r in failed(v, "recap_claims"))


def test_a_team_passing_phrase_needs_that_teams_claim_even_with_the_star_claim():
    v = check(HT, claims_used=[c for c in HT.out.claims_used if c != "passing:Harbourne Athletic"])
    assert any("says how ['Harbourne Athletic'] passed without the passing claim" in r
               for r in failed(v, "recap_claims"))
