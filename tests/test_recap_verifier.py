import pytest

from xplain import verifier
from xplain.prompts import RECAP_EXAMPLES, recap_instructions
from xplain.recap import recap_claims
from xplain.recap_verifier import RECAP_WORD_LIMIT, verify_recap

FT, HT = RECAP_EXAMPLES
PLAYERS = ["Dario Pell", "Joss Marlin", "Tomas Avery", "Ollie Branck"]


def check(example, players=PLAYERS, **update):
    return verify_recap(example.out.model_copy(update=update), example.card, players)


def sub(example, lang, old, new):
    text = getattr(example.out, f"text_{lang}")
    assert old in text, old
    return {f"text_{lang}": text.replace(old, new)}


def failed(verdict, check_name):
    return [r for r in verdict.reasons if r.startswith(check_name)]


@pytest.mark.parametrize("example", RECAP_EXAMPLES, ids=lambda e: e.name)
def test_the_examples_pass(example):
    verdict = verify_recap(example.out, example.card, PLAYERS)
    assert verdict.ok, verdict.reasons
    assert set(example.out.claims_used) <= set(recap_claims(example.card))


def test_the_brief_holds_the_examples_and_the_limits():
    brief = recap_instructions()
    assert FT.out.text_el in brief and HT.out.text_en in brief
    assert f"{RECAP_WORD_LIMIT} words" in brief


def test_numbers_and_number_words_are_rejected():
    assert failed(check(FT, **sub(FT, "en", "came from behind", "came from behind 2-1")), "no_numbers")
    assert failed(check(FT, **sub(FT, "es", "un gol tardío", "dos goles, uno tardío")), "no_numbers")
    assert failed(check(FT, **sub(FT, "el", "στο φινάλε", "στο φινάλε μετά από δύο λεπτά")), "no_numbers")
    assert failed(check(FT, **sub(FT, "en", "late winner", "winner in the ninetieth minute")), "no_numbers")


def test_format_and_length():
    long_text = FT.out.text_en.replace("opener.", "opener " + "and more " * 10 + ".")
    assert failed(check(FT, text_en=long_text), "length")
    three = FT.out.text_en + " Both teams were applauded."
    assert failed(check(FT, text_en=three), "format")
    assert failed(check(FT, text_es=""), "format")


def test_judgement_luck_and_model_words_are_banned():
    assert failed(check(FT, **sub(FT, "en", "came from behind to beat", "deservedly came from behind to beat")),
                  "banned")
    assert failed(check(FT, **sub(FT, "es", "remontó y venció", "remontó con suerte y venció")), "banned")
    assert failed(check(FT, **sub(FT, "el", "έκανε την ανατροπή", "έκανε την ανατροπή κυριαρχώντας")), "banned")
    assert failed(check(FT, **sub(FT, "en", "completed more passes than expected",
                                  "completed more passes than the xP model expected")), "banned")


def test_no_cause_in_a_recap():
    v = check(FT, **sub(FT, "en", "with Tomas Avery's late winner", "because of Tomas Avery's late winner"))
    assert failed(v, "recap_cause")
    v = check(FT, **sub(FT, "el", "μετά την ισοφάριση", "επειδή είχε προηγηθεί η ισοφάριση"))
    assert failed(v, "recap_cause")


def test_unknown_and_duplicate_claim_ids():
    assert failed(check(FT, claims_used=FT.out.claims_used + ["comeback:Kellsworth Rovers"]), "recap_claims")
    assert failed(check(FT, claims_used=FT.out.claims_used + ["result"]), "recap_claims")


def test_a_used_claim_must_be_expressed_in_every_language():
    v = check(FT, **sub(FT, "es", "remontó y venció", "venció"))
    assert any("comeback" in r and "text_es" in r for r in failed(v, "recap_claims"))
    v = check(FT, **sub(FT, "el", "του Dario Pell", "της αντίπαλης ομάδας"))
    assert any("scorer:Dario Pell" in r and "text_el" in r for r in failed(v, "recap_claims"))


def test_story_words_need_their_claim():
    no_comeback = [c for c in FT.out.claims_used if not c.startswith("comeback")]
    assert any("comeback" in r for r in failed(check(FT, claims_used=no_comeback), "recap_claims"))
    no_late = [c for c in FT.out.claims_used if not c.startswith(("late_", "scorer:Tomas"))]
    assert any("late" in r for r in failed(check(FT, claims_used=no_late), "recap_claims"))
    no_passing = [c for c in HT.out.claims_used if not c.startswith("passing")]
    assert any("passing" in r for r in failed(check(HT, claims_used=no_passing), "recap_claims"))


def test_the_result_words_must_fit_the_result():
    v = check(FT, **sub(FT, "en", "came from behind to beat Kellsworth Rovers",
                        "came from behind to draw with Kellsworth Rovers"))
    assert any("was won" in r for r in failed(v, "recap_claims"))
    v = check(HT, **sub(HT, "en", "are level and goalless at the break", "won the first half"))
    assert any("half time" in r for r in failed(v, "recap_claims"))
    v = check(FT, **sub(FT, "en", "Harbourne Athletic came from behind to beat Kellsworth Rovers",
                        "Harbourne Athletic came from behind, Kellsworth Rovers won"))
    assert any("gives the win or lead to Kellsworth Rovers" in r for r in failed(v, "recap_claims"))


def test_passing_phrases_name_the_right_team_in_their_clause():
    v = check(FT, **sub(FT, "en", "Harbourne Athletic completed more passes than expected; Kellsworth Rovers "
                                  "completed fewer than expected",
                        "Kellsworth Rovers completed more passes than expected; Harbourne Athletic completed fewer "
                        "than expected"))
    assert any("'above' expected" in r for r in failed(v, "recap_claims"))
    v = check(HT, **sub(HT, "el", "η Harbourne Athletic κινήθηκε κοντά στα αναμενόμενα",
                        "η Harbourne Athletic ολοκλήρωσε περισσότερες πάσες από τις αναμενόμενες"))
    assert any("'above' expected" in r and "text_el" in r for r in failed(v, "recap_claims"))
    v = check(FT, **sub(FT, "en", "; Kellsworth Rovers completed fewer than expected", ", fewer than expected"))
    assert any("no team in its clause" in r for r in failed(v, "recap_claims"))


def test_only_claimed_scorers_are_named():
    v = check(FT, **sub(FT, "en", "Harbourne Athletic completed more", "Ollie Branck and Harbourne Athletic completed more"))
    assert any("names Ollie Branck" in r for r in failed(v, "recap_claims"))
    v = check(FT, claims_used=[c for c in FT.out.claims_used if c != "scorer:Joss Marlin"])
    assert any("names Joss Marlin" in r for r in failed(v, "recap_claims"))


def test_the_narrator_verifier_tables_are_read_not_changed():
    # recap_verifier only reads verifier's tables; the Narrator's limits stay as they are
    assert verifier.WORD_LIMIT == 24 and "\\bsuccess\\w*" in verifier.BANNED["en"]


WON_ES = "Harbourne Athletic remontó y venció a Kellsworth Rovers"
EQUALISER_ES = "después de que Joss Marlin igualara el tanto inicial de Dario Pell"


@pytest.mark.parametrize("draw", ["acabó en empate", "el partido terminó con un empate", "hubo empate"])
def test_spanish_empate_in_a_won_match_is_a_draw_word(draw):
    v = check(FT, **sub(FT, "es", WON_ES, f"{WON_ES}, aunque {draw}"))
    assert any("text_es has" in r and "but the match was won" in r for r in failed(v, "recap_claims")), v.reasons


@pytest.mark.parametrize("equaliser", [
    "después de que Joss Marlin lograra el empate tras el tanto inicial de Dario Pell",
    "tras el tanto inicial de Dario Pell y el gol del empate de Joss Marlin",
    "tras el tanto inicial de Dario Pell y el empate de Joss Marlin",
    "tras el tanto inicial de Dario Pell y el empate de Marlin",
    "después de que Joss Marlin consiguiera el empate tras el tanto inicial de Dario Pell",
    "después de que Joss Marlin pusiera el empate tras el tanto inicial de Dario Pell",
])
def test_spanish_equaliser_phrases_are_allowed_in_a_won_match(equaliser):
    v = check(FT, **sub(FT, "es", EQUALISER_ES, equaliser))
    assert v.ok, v.reasons


def test_logro_el_empate_is_not_a_draw_but_logro_stays_a_banned_credit_verb():
    text = FT.out.text_es.replace(EQUALISER_ES, "y antes Joss Marlin logró el empate tras el tanto inicial de Dario Pell")
    v = check(FT, text_es=text)
    assert failed(v, "recap_claims") == []                   # the draw check allows the equaliser phrase
    assert v.reasons == ["banned: text_es has 'logró'"]      # the credit-verb ban (verifier.BANNED) still applies


def test_empate_of_someone_who_is_not_a_scorer_is_still_a_draw_word():
    v = check(FT, **sub(FT, "es", EQUALISER_ES, "tras el tanto inicial de Dario Pell y el empate de ambos equipos"))
    assert any("but the match was won" in r for r in failed(v, "recap_claims"))


DRAW_CARD = FT.card.model_copy(update={
    "score_home": 1, "score_away": 1, "result": "draw", "comeback_team_id": None, "late_winner_team_id": None,
    "goals": [FT.card.goals[1].model_copy(update={"period": 1, "home": 1, "away": 0}),
              FT.card.goals[0].model_copy(update={"period": 2, "home": 1, "away": 1})],
    "home": FT.card.home.model_copy(update={"band": "close"}), "away": FT.card.away.model_copy(update={"band": "close"}),
})
DRAW_OUT = FT.out.model_copy(update={
    "text_en": "Harbourne Athletic and Kellsworth Rovers drew, with goals from Joss Marlin and Dario Pell. "
               "Harbourne Athletic passed about as expected, Kellsworth Rovers passed about as expected.",
    "text_es": "Harbourne Athletic y Kellsworth Rovers empataron, con goles de Joss Marlin y Dario Pell. "
               "Harbourne Athletic estuvo en línea con lo esperado, Kellsworth Rovers estuvo en línea con lo esperado.",
    "text_el": "Η Harbourne Athletic και η Kellsworth Rovers έμειναν στην ισοπαλία, με γκολ των Joss Marlin και "
               "Dario Pell. Η Harbourne Athletic κινήθηκε κοντά στα αναμενόμενα, η Kellsworth Rovers κινήθηκε "
               "κοντά στα αναμενόμενα.",
    "claims_used": ["result", "scorer:Joss Marlin", "scorer:Dario Pell", "passing:Harbourne Athletic",
                    "passing:Kellsworth Rovers"],
})


def test_a_real_draw_still_needs_a_draw_word():
    good = verify_recap(DRAW_OUT, DRAW_CARD, PLAYERS)
    assert good.ok, good.reasons
    only_equaliser = DRAW_OUT.text_es.replace(
        "Harbourne Athletic y Kellsworth Rovers empataron, con goles de Joss Marlin y Dario Pell",
        "Joss Marlin marcó para Harbourne Athletic y Dario Pell anotó el gol del empate para Kellsworth Rovers")
    v = verify_recap(DRAW_OUT.model_copy(update={"text_es": only_equaliser}), DRAW_CARD, PLAYERS)
    assert "recap_claims: text_es does not express the result claim" in v.reasons
    bare = DRAW_OUT.text_es.replace("empataron", "terminaron con empate")
    assert verify_recap(DRAW_OUT.model_copy(update={"text_es": bare}), DRAW_CARD, PLAYERS).ok


def test_the_brief_lists_the_late_words_and_the_greek_after():
    brief = recap_instructions()
    assert "en el tiempo añadido" in brief and "στις καθυστερήσεις" in brief and "μετά" in brief
