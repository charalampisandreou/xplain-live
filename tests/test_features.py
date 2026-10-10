import json

import numpy as np
import pytest
from pydantic import ValidationError

from xplain.features import (
    FEATURE_ORDER,
    FeatureOrderError,
    angle_to_goal,
    build_features,
    validate_feature_order,
)
from xplain.schema import (
    BodyPart,
    PassDetails,
    PassEvent,
    PassHeight,
    PassInputs,
    PlayPattern,
)


def make_inputs(**kw) -> PassInputs:
    base = dict(
        start_x=60.0,
        start_y=40.0,
        height=PassHeight.GROUND,
        body_part=BodyPart.FOOT,
        play_pattern=PlayPattern.REGULAR,
        under_pressure=False,
    )
    base.update(kw)
    return PassInputs(**base)


def test_feature_order_has_17_unique_columns():
    assert len(FEATURE_ORDER) == 17 and len(set(FEATURE_ORDER)) == 17


def test_no_end_of_pass_columns():
    assert {"pass_angle", "pass_length", "end_x", "end_y"}.isdisjoint(FEATURE_ORDER)


def test_angle_matches_training_formula():
    # same expectations as xPass-Project tests/test_features.py
    got = angle_to_goal(np.array([60.0, 120.0, 100.0]), np.array([40.0, 0.0, 60.0]))
    np.testing.assert_allclose(got, [0.0, np.pi / 2, np.arctan2(-20, 20)])


def test_build_features_columns_and_values():
    df = build_features(
        [
            make_inputs(),
            make_inputs(
                start_x=100.0, start_y=60.0, height=PassHeight.HIGH,
                body_part=BodyPart.HEAD, play_pattern=PlayPattern.CORNER, under_pressure=True,
            ),
            make_inputs(body_part=BodyPart.OTHER, play_pattern=PlayPattern.OTHER),
        ]
    )
    assert list(df.columns) == FEATURE_ORDER
    r = df.iloc[1]
    assert r["height_High Pass"] == 1 and r["height_Low Pass"] == 0
    assert r["body_part_Head"] == 1 and r["body_part_Foot"] == 0
    assert r["play_pattern_Corner"] == 1 and r["under_pressure"] == 1
    assert r["dist_to_goal"] == pytest.approx(np.hypot(20, 20))
    # baselines: ground pass, "Other" body part and play pattern -> all dummy columns zero
    b = df.iloc[2]
    dummies = [c for c in FEATURE_ORDER if c.startswith(("height_", "body_part_", "play_pattern_"))]
    assert b[[c for c in dummies if c.startswith("body_part_")]].sum() == 0
    assert b[[c for c in dummies if c.startswith("play_pattern_")]].sum() == 0
    assert b[[c for c in dummies if c.startswith("height_")]].sum() == 0


def test_build_features_rejects_non_input_types():
    details = PassDetails(end_x=70, end_y=40, length=10, ball_speed_ms=15, completed=True)
    with pytest.raises(TypeError):
        build_features([details])  # end-derived fields can't reach the model


def test_schema_separates_inputs_from_details():
    assert set(PassInputs.model_fields).isdisjoint(PassDetails.model_fields)
    assert {"end_x", "end_y", "length", "ball_speed_ms", "completed"}.isdisjoint(PassInputs.model_fields)


def test_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        make_inputs(start_x=121.0)
    with pytest.raises(ValidationError):
        PassInputs(start_x=1, start_y=1, height="Ground Pass", body_part="Foot",
                   play_pattern="Regular Play", under_pressure=False, end_x=5)  # extra forbidden


def test_pass_event_roundtrip():
    ev = PassEvent(
        event_id="e1", match_id="m1", seq=0, t=1.5, period=1, team_id="t1", player_id="p1",
        inputs=make_inputs(),
        details=PassDetails(end_x=70, end_y=42, length=10.2, ball_speed_ms=14.0, completed=True),
    )
    again = PassEvent.model_validate_json(ev.model_dump_json())
    assert again == ev


def test_validate_feature_order(tmp_path):
    good = tmp_path / "feature_list.json"
    good.write_text(json.dumps(FEATURE_ORDER))
    assert validate_feature_order(good) == FEATURE_ORDER

    swapped = list(FEATURE_ORDER)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(swapped))
    with pytest.raises(FeatureOrderError, match="different order"):
        validate_feature_order(bad)

    old = tmp_path / "old.json"
    old.write_text(json.dumps(["pass_angle" if c == "angle_to_goal" else c for c in FEATURE_ORDER]))
    with pytest.raises(FeatureOrderError, match="pass_angle"):
        validate_feature_order(old)
