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


def test_spanish_empate_as_the_equaliser_is_not_a_draw():
    ok = check(FT, **sub(FT, "es", "igualara el tanto inicial de Dario Pell",
                         "marcara el empate tras el tanto inicial de Dario Pell"))
    assert ok.ok, ok.reasons
    v = check(FT, **sub(FT, "es", "Harbourne Athletic remontó y venció a Kellsworth Rovers",
                        "Harbourne Athletic remontó y el partido terminó en empate con Kellsworth Rovers"))
    assert any("was won" in r for r in failed(v, "recap_claims"))


def test_the_brief_lists_the_late_words_and_the_greek_after():
    brief = recap_instructions()
    assert "en el tiempo añadido" in brief and "στις καθυστερήσεις" in brief and "μετά" in brief
