import itertools
import re

import pytest

from xplain.contracts import Driver
from xplain.explainer import UNITS, depth_zone, unit_value_and_label, width_zone
from xplain.phrases import (
    DEPTH_PHRASES,
    LABEL_PHRASES,
    NOT_NARRATED,
    WIDTH_PHRASES,
    driver_phrases,
    position_phrase,
)
from xplain.schema import PITCH_LENGTH, PITCH_WIDTH, BodyPart, PassHeight, PassInputs, PlayPattern
from xplain.verifier import BANNED, EASY_WORDS, HARD_WORDS, LANGS, NUMBER_WORDS


def _inputs(x: float = 60.0, y: float = 40.0, height=PassHeight.GROUND, body=BodyPart.FOOT,
            pattern=PlayPattern.REGULAR, pressure: bool = False) -> PassInputs:
    return PassInputs(start_x=x, start_y=y, height=height, body_part=body, play_pattern=pattern,
                      under_pressure=pressure)


def emitted_drivers() -> list[Driver]:
    """Every driver the Explainer can emit: each unit over every enum value, positions over a 1 m grid."""
    inputs = [_inputs(height=h, body=b, pattern=p, pressure=u)
              for h, b, p, u in itertools.product(PassHeight, BodyPart, PlayPattern, (True, False))]
    inputs += [_inputs(x=float(x), y=float(y))
               for x in range(0, int(PITCH_LENGTH) + 1) for y in range(0, int(PITCH_WIDTH) + 1)]
    seen: dict[tuple[str, str], Driver] = {}
    for unit in UNITS:
        for i in inputs:
            value, label = unit_value_and_label(unit, i)
            seen.setdefault((unit.name, label), Driver(rank=1, feature=unit.name, topic=unit.topic, value=value,
                                                       label=label, shap=0.1, direction="raises"))
    return list(seen.values())


def test_every_emitted_label_has_a_phrase_in_all_three_languages():
    drivers = emitted_drivers()
    assert len(drivers) == 3 + 4 + 8 + 2 + 12   # heights, body parts, play patterns, pressure, positions
    for d in drivers:
        for piece in driver_phrases(d).values():
            assert set(piece) == set(LANGS) and all(piece[lang].strip() for lang in LANGS), d.label


def test_the_tables_have_no_stale_entries():
    drivers = emitted_drivers()
    assert set(LABEL_PHRASES) == {d.label for d in drivers if d.feature != "position"}
    xs, ys = range(0, int(PITCH_LENGTH) + 1), range(0, int(PITCH_WIDTH) + 1)
    assert set(DEPTH_PHRASES) == {depth_zone(float(x)) for x in xs}
    assert set(WIDTH_PHRASES) == {width_zone(float(y)) for y in ys}


def test_position_phrase_is_depth_then_width():
    assert position_phrase("near_goal", "out_wide") == {
        "en": "close to the opposition goal, out wide",
        "es": "cerca de la portería rival, desde la banda",
        "el": "κοντά στην αντίπαλη εστία, από τα πλάγια",
    }
    d = Driver(rank=2, feature="position", topic="pitch position", value="defensive_third|central",
               label="from their own defensive third, centrally", shap=0.3, direction="raises")
    assert driver_phrases(d) == {"phrase": position_phrase("defensive_third", "central"),
                                 "depth": DEPTH_PHRASES["defensive_third"], "width": WIDTH_PHRASES["central"]}


def test_not_narrated_labels():
    assert NOT_NARRATED == {"with the foot", "in open play", "with no pressure on the passer",
                            "with another part of the body", "from another kind of play"}
    assert NOT_NARRATED <= set(LABEL_PHRASES)


def narrated_phrases() -> list[tuple[str, str]]:
    """(lang, phrase) for every phrase the model can be given: composed positions, their pieces and labels."""
    out = []
    for lang in LANGS:
        out += [(lang, position_phrase(d, w)[lang]) for d in DEPTH_PHRASES for w in WIDTH_PHRASES]
        out += [(lang, p[lang]) for p in (*DEPTH_PHRASES.values(), *WIDTH_PHRASES.values())]
        out += [(lang, p[lang]) for label, p in LABEL_PHRASES.items() if label not in NOT_NARRATED]
    return out


@pytest.mark.parametrize("lang, phrase", narrated_phrases())
def test_no_banned_number_or_difficulty_pattern_fires_on_an_approved_phrase(lang, phrase):
    for table in (BANNED, NUMBER_WORDS, EASY_WORDS, HARD_WORDS):
        for pattern in table[lang]:
            assert not re.search(pattern, phrase, re.IGNORECASE), (pattern, phrase)


@pytest.mark.parametrize("label", ["with another part of the body", "from another kind of play"])
def test_the_vague_labels_are_banned_in_every_language(label):
    for lang in LANGS:
        assert any(re.search(p, LABEL_PHRASES[label][lang], re.IGNORECASE) for p in BANNED[lang]), lang
