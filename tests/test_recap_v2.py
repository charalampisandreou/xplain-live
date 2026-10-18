"""Recap v2: the star player, the slots, and the recap_slots checks on the raw text."""
from xplain.prompts import RECAP_EXAMPLES
from xplain.recap import (STAR_MIN_PVA, half_up, recap_claims, recap_facts, slot_directions, slot_values,
                          star_player)
from xplain.recap_verifier import RECAP_WORD_LIMIT, verify_recap

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_recap import card, marks  # noqa: F401  (fixture)

FT, HT, DRAW_FT = RECAP_EXAMPLES
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
        assert recap_facts(c)["star"] == {"team": c.star.team_name}


def test_slots_exist_only_for_above_or_below_teams_and_a_star():
    assert slot_values(FT.card) == {"home_vs_expected": "15", "away_vs_expected": "15", "star_name": "Ollie Branck",
                                    "star_pva": "6"}
    assert slot_directions(FT.card) == {"home_vs_expected": "more", "away_vs_expected": "fewer", "star_pva": "more"}
    assert slot_values(HT.card) == {"away_vs_expected": "15", "star_name": "Dario Pell", "star_pva": "4"}
    assert slot_values(DRAW_FT.card) == {"home_vs_expected": "15"}   # away is close, no star
    assert set(recap_facts(DRAW_FT.card)["slots"]) == {"home_vs_expected"}


# --- recap_slots and the filled text ---------------------------------------------------------------------

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
