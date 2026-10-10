import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_variants as cv  # noqa: E402

from xplain.config import GenConfig  # noqa: E402
from xplain.generator import generate_match  # noqa: E402
from xplain.schema import PassEvent  # noqa: E402


def test_variant_config_maps_fields_and_zeroes_missing_terms():
    cfg = cv.variant_config({"intercept": 2.3, "is_high": -2.6, "keeper": 0.2})  # start-only shape
    assert (cfg.intercept, cfg.w_high, cfg.bonus_keeper) == (2.3, -2.6, 0.2)
    assert cfg.w_length == 0.0 and cfg.w_forward == 0.0 and cfg.w_start_x == 0.0
    assert cfg.w_skill == GenConfig().w_skill


def test_logged_match_keeps_events_and_flags_restarts():
    m = cv._LoggedMatch(42, GenConfig())
    m.run()
    passes = [e for e in m.build_events() if isinstance(e, PassEvent)]
    reference = [e for e in generate_match(42)[0] if isinstance(e, PassEvent)]
    # logging adds no random draws; seq/event_id differ only because player_load events are skipped
    drop = {"seq", "event_id"}
    assert [p.model_dump(exclude=drop) for p in passes] == [p.model_dump(exclude=drop) for p in reference]
    assert len(m.restart_log) == len(passes)
    assert 0 < sum(m.restart_log) < len(passes)
    assert passes[0].inputs.play_pattern == "Kick Off" and m.restart_log[0]


def test_calibration_slope_is_one_for_calibrated_probabilities():
    import numpy as np

    rng = np.random.default_rng(3)
    xp = rng.uniform(0.05, 0.95, 50_000)
    y = (rng.random(xp.size) < xp).astype(float)
    c = cv.calibration(xp, y)
    assert abs(c["slope"] - 1.0) < 0.05
    assert c["brier"] == np.mean((xp - y) ** 2)


def test_logged_terms_sum_to_generator_logit():
    # _LoggedMatch raises if the logged terms do not reproduce the logit the generator used
    m = cv._LoggedMatch(7, GenConfig())
    m.run()
    assert len(m.term_log) == len(m.restart_log)
    assert set(m.term_log[0]) == set(cv.TERMS)
