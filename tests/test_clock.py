import math

import pytest
from pydantic import ValidationError

from xplain.clock import HALF_LENGTH_S, MatchClock, as_of_t, match_clock, second_half_start
from xplain.config import GenConfig
from xplain.generator import generate_match
from xplain.narration import NarratorOut
from xplain.pipeline import process_match
from xplain.schema import PassEvent
from xplain.verifier import verify

from .test_pipeline import seed42_events, tiny_scorer, tiny_model  # noqa: F401  (fixtures)
from .test_verifier import make_bundle

START = 2844.15  # seed 42: the second-half kick-off (first period-2 event)


def label(t: float, period: int = 1, start: float | None = START) -> str:
    return match_clock(t, period, start).label


def test_half_length_is_read_from_the_generator():
    assert HALF_LENGTH_S == GenConfig().half_length_s == 2700.0


@pytest.mark.parametrize("t, expected", [(0.0, "1"), (59.9, "1"), (60.0, "2"), (2640.0, "45"), (2699.99, "45")])
def test_first_half_regular_time(t, expected):
    assert label(t) == expected


@pytest.mark.parametrize("t, minute, added, expected", [
    (2700.0, 45, 1, "45+1"), (2759.99, 45, 1, "45+1"), (2760.0, 45, 2, "45+2"), (2938.0, 45, 4, "45+4"),
])
def test_first_half_added_time(t, minute, added, expected):
    assert match_clock(t, 1, None) == MatchClock(minute=minute, added=added, label=expected)


def test_first_pass_of_second_half_is_46():
    assert match_clock(START, 2, START) == MatchClock(minute=46, added=0, label="46")
    assert label(START + 59.9, 2) == "46"
    assert label(START + 60.0, 2) == "47"


@pytest.mark.parametrize("offset, expected", [
    (2699.99, "90"), (2700.0, "90+1"), (2700.0 + 59.99, "90+1"), (2700.0 + 300.0, "90+6"),
])
def test_second_half_added_time(offset, expected):
    assert label(START + offset, 2) == expected


def test_previous_bug_case_second_half_was_too_high():
    # seed 42 has 2:24 of first-half added time; t = 4020 used to show floor(4020 / 60) + 1 = 68
    t = 4020.0
    assert math.floor(t / 60) + 1 == 68
    assert match_clock(t, 2, START).minute == 65   # 45 + floor((4020 - 2844.15) / 60) + 1: 3 lower


def test_float_noise_does_not_move_a_boundary():
    assert START + 2700.0 - START < 2700.0          # the raw subtraction falls just short
    assert label(START + 2700.0, 2) == "90+1"


def test_period_two_needs_a_start_at_or_before_t():
    with pytest.raises(ValueError, match="second_half_start"):
        match_clock(3000.0, 2, None)
    with pytest.raises(ValueError, match="second_half_start"):
        match_clock(3000.0, 2, 3100.0)
    with pytest.raises(ValueError, match="period"):
        match_clock(10.0, 3, START)


def test_match_clock_is_frozen():
    with pytest.raises(ValidationError):
        match_clock(0.0, 1, None).minute = 2


def test_second_half_start_is_the_first_period_two_event(seed42_events):  # noqa: F811
    assert second_half_start(seed42_events) == START
    assert second_half_start([e for e in seed42_events if e.period == 1]) is None


def test_as_of_t_uses_the_clock(seed42_events):  # noqa: F811
    p1_last = max(e.t for e in seed42_events if e.period == 1)
    assert as_of_t(seed42_events, 0) == -math.inf
    assert as_of_t(seed42_events, 45) == p1_last                 # minute 45 includes the first half's added time
    assert as_of_t(seed42_events, 46) < START + 60.0
    assert as_of_t(seed42_events, 90) == max(e.t for e in seed42_events)


# --- seed 42 ------------------------------------------------------------------------------------

def test_seed42_minutes_never_above_90_and_never_decrease(tiny_scorer, seed42_events):  # noqa: F811
    from .test_pipeline import WIDE
    bundles = process_match(seed42_events, tiny_scorer, WIDE)
    assert {b.moment.period for b in bundles} == {1, 2}
    start = second_half_start(seed42_events)
    for period, cap in ((1, 45), (2, 90)):
        clocks = [(b.moment.minute, b.moment.minute_added) for b in bundles if b.moment.period == period]
        assert clocks == sorted(clocks)
        assert all(minute <= cap for minute, _ in clocks)
        assert all(added == 0 or minute == cap for minute, added in clocks)
    passes = [e for e in seed42_events if isinstance(e, PassEvent)]
    for period in (1, 2):
        clocks = [match_clock(p.t, p.period, start) for p in passes if p.period == period]
        assert [(c.minute, c.added) for c in clocks] == sorted((c.minute, c.added) for c in clocks)
    assert max(match_clock(p.t, p.period, start).minute for p in passes) == 90
    assert min(match_clock(p.t, p.period, start).minute for p in passes if p.period == 2) == 46


def test_other_seeds_have_a_second_half_start_matching_the_truth():
    for seed in (1, 7):
        events, truth = generate_match(seed)
        assert second_half_start(events) == truth["periods"]["p2_start_t"]


# --- verifier with added time --------------------------------------------------------------------

def test_added_time_label_in_a_line_is_rejected():
    # the card shows minute_label ("45+2"); the lines never carry the minute
    bundle = make_bundle(minute=45, added=2)
    out = NarratorOut(line_en="In 45+2, Wendez's easy pass goes astray.",
                      line_es="En el tiempo añadido, el pase fácil de Wendez se pierde.",
                      line_el="Στις καθυστερήσεις, η εύκολη πάσα του Wendez χάνεται.", drivers_used=[])
    assert verify(out, bundle).reasons == ["no_numbers: line_en has '45'", "no_numbers: line_en has '2'"]
