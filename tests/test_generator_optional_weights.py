import hashlib
from dataclasses import replace

import pytest

from xplain.config import GenConfig
from xplain.generator import generate_match
from xplain.schema import MatchMetaEvent

LOCATION_FIELDS = ("w_start_x", "w_dist_goal", "w_angle_goal")

# The hand-set completion weights of generator 0.1.0, with the location terms off.
OLD_WEIGHTS = dict(
    intercept=2.9, w_skill=0.45, w_length=-0.045, w_forward=-0.03, w_pressure=-0.55, w_high=-0.35,
    w_low=-0.25, bonus_kick_off=1.5, bonus_throw_in=0.3, bonus_goal_kick=0.0, bonus_free_kick=0.5,
    bonus_corner=-0.5, bonus_keeper=0.5, w_start_x=0.0, w_dist_goal=0.0, w_angle_goal=0.0,
)
OLD_CONFIG = replace(GenConfig(), **OLD_WEIGHTS)
OLD_VERSION = "0.1.0"

# SHA-256 of the events JSONL from generator 0.1.0 (default config at the time), recorded before
# the location weights were added.
GOLDEN = {
    1: "75248959e8b1700013972baed87bf4cb3f1a9a098125a9a965b05574390a809c",
    7: "621a2914900b3042673b9b9fba632531f6a4a71a5c3fa570f3613ff551460dc7",
    42: "064e35dbc5e1976b43701e134fda125f15face85ee35afde7aea9f0dbd8efb0a",
}


def _events_hash(seed: int, cfg: GenConfig, as_version: str | None = None) -> str:
    """Hash of the events JSONL; as_version rewrites the version stamp in match_meta only."""
    events, _ = generate_match(seed, cfg)
    if as_version is not None:
        events = [e.model_copy(update={"generator_version": as_version}) if isinstance(e, MatchMetaEvent) else e
                  for e in events]
    return hashlib.sha256("".join(e.model_dump_json() + "\n" for e in events).encode()).hexdigest()


@pytest.mark.parametrize("seed", sorted(GOLDEN))
def test_old_weights_reproduce_pre_change_events(seed):
    # everything except the match_meta version stamp must be byte-identical to generator 0.1.0
    assert _events_hash(seed, OLD_CONFIG, as_version=OLD_VERSION) == GOLDEN[seed]


@pytest.mark.parametrize("name", LOCATION_FIELDS)
def test_location_weight_changes_outcomes(name):
    # the terms are wired into the logit: a large weight must change some pass outcomes
    assert _events_hash(42, replace(OLD_CONFIG, **{name: -0.5})) != _events_hash(42, OLD_CONFIG)


def test_defaults_are_fitted_variant_c():
    cfg = GenConfig()
    assert cfg.intercept == pytest.approx(9.19253)
    assert cfg.w_skill == 0.45
    assert all(getattr(cfg, name) != 0.0 for name in LOCATION_FIELDS)
