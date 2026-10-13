import hashlib
from dataclasses import replace

import pytest

from xplain.config import GENERATOR_VERSION, GenConfig
from xplain.generator import generate_match

# The designed completion rule of generator 0.2.1 (config.GenConfig): round values, signs from football
# reasoning, not fitted to any data. 0.2.1 scaled the context terms of 0.2.0 (position and pressure x2, height
# and head x1.5) and re-solved the intercept for about 78% completion.
DESIGNED = dict(
    intercept=4.6, w_skill=0.45, w_x=-2.0, w_y=-0.8, w_pressure=-1.2, w_high=-1.2, w_low=-0.6, w_head=-1.05,
    w_length=-0.04, w_forward=-0.02, bonus_kick_off=2.0, bonus_throw_in=0.8, bonus_goal_kick=-0.3,
    bonus_free_kick=0.5, bonus_corner=-1.0, bonus_keeper=0.5,
)

# SHA-256 of the events JSONL from generator 0.2.1 with the default config, recorded when the wider designed
# spread replaced 0.2.0. Regenerate only on a deliberate generator change (and bump GENERATOR_VERSION).
GOLDEN = {
    1: "c54fe671159043482f51796aa295c5d1a85ebf933dfd83ea5920052a39ab7b5b",
    7: "0aea884fde21b7f4af615a0ef626b82e297d044dc3c2eb276e92802acdb0fe49",
    42: "f5496e0376c6cb80236effe1fca8a99606b59ae520e01588ff7de3d583098ebf",
}


def _events_hash(seed: int, cfg: GenConfig = GenConfig()) -> str:
    events, _ = generate_match(seed, cfg)
    return hashlib.sha256("".join(e.model_dump_json() + "\n" for e in events).encode()).hexdigest()


def test_defaults_are_the_designed_rule():
    cfg = GenConfig()
    assert {name: getattr(cfg, name) for name in DESIGNED} == DESIGNED
    assert GENERATOR_VERSION == "0.2.1"


def test_old_fitted_fields_are_gone():
    for name in ("w_start_x", "w_dist_goal", "w_angle_goal"):
        assert not hasattr(GenConfig(), name)


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_default_events_are_byte_identical(seed):
    assert _events_hash(seed) == GOLDEN[seed]


@pytest.mark.parametrize("name", ["w_x", "w_y", "w_pressure", "w_high", "w_low", "w_head", "w_length",
                                  "w_forward", "bonus_kick_off", "bonus_corner"])
def test_every_term_is_wired_into_the_rule(name):
    # a large change to any one weight must change some pass outcomes in at least one golden match (a rare term,
    # such as the corner bonus, may not flip an outcome in every single match)
    cfg = replace(GenConfig(), **{name: getattr(GenConfig(), name) - 3.0})
    assert any(_events_hash(seed, cfg) != GOLDEN[seed] for seed in sorted(GOLDEN))


@pytest.mark.parametrize("name", ["w_x", "w_y", "w_pressure", "w_high", "w_low", "w_head", "w_length", "w_forward"])
def test_difficulty_terms_make_passes_harder(name):
    assert getattr(GenConfig(), name) < 0


def test_restart_order():
    cfg = GenConfig()
    assert cfg.bonus_kick_off > cfg.bonus_throw_in > 0 > cfg.bonus_corner
    assert cfg.bonus_corner == min(cfg.bonus_kick_off, cfg.bonus_throw_in, cfg.bonus_goal_kick,
                                   cfg.bonus_free_kick, cfg.bonus_corner, cfg.bonus_keeper)
