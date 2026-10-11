import json

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from xplain.analyst import build_moment
from xplain.contracts import Driver, Explanation, Moment
from xplain.explainer import (
    ADDITIVITY_TOLERANCE,
    BODY_PART_LABELS,
    DEPTH_ATTACKING_FROM,
    DEPTH_MIDDLE_FROM,
    DEPTH_NEAR_GOAL_FROM,
    HEIGHT_LABELS,
    PLAY_PATTERN_LABELS,
    POSITION_LABELS,
    PRESSURE_LABELS,
    UNIT_SPECS,
    UNITS,
    WIDTH_OFF_CENTRE_FROM,
    WIDTH_OUT_WIDE_FROM,
    ZERO_SHAP,
    ExplanationError,
    ShapExplainer,
    build_explanation_evidence,
    build_units,
    depth_zone,
    drivers_and_rest,
    position_value_and_label,
    rank_units,
    unit_shap,
    width_zone,
)
from xplain.features import FEATURE_ORDER, build_features
from xplain.generator import generate_match
from xplain.schema import PITCH_LENGTH, PITCH_WIDTH, BodyPart, PassDetails, PassEvent, PassHeight, PassInputs, PlayPattern
from xplain.scoring import ModelLoadError, XPassScorer, find_default_model

N_FOLDS = 3
GROUPS = ("height", "body_part", "play_pattern")
UNIT_NAMES = [u.name for u in UNITS]


def random_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random rows with the 17 model columns: numeric start/goal features, 0/1 for the rest."""
    df = pd.DataFrame(rng.integers(0, 2, size=(n, len(FEATURE_ORDER))), columns=FEATURE_ORDER).astype(float)
    df["start_x"] = rng.uniform(0, 120, n)
    df["start_y"] = rng.uniform(0, 80, n)
    df["angle_to_goal"] = np.arctan2(40 - df["start_y"], 120 - df["start_x"])
    df["dist_to_goal"] = np.hypot(120 - df["start_x"], 40 - df["start_y"])
    return df


@pytest.fixture(scope="module")
def tiny_model() -> CalibratedClassifierCV:
    """Small CalibratedClassifierCV over XGBClassifier, trained on random data with a fixed seed."""
    rng = np.random.default_rng(7)
    X = random_features(600, rng)
    logit = (2.5 - 0.03 * X["start_x"] - 1.2 * X["under_pressure"] - 1.5 * X["height_High Pass"]
             - 1.0 * X["body_part_Head"] + 0.8 * X["play_pattern_Throw In"])
    y = (rng.uniform(size=len(X)) < 1 / (1 + np.exp(-logit))).astype(int)
    base = XGBClassifier(n_estimators=8, max_depth=3, learning_rate=0.3, n_jobs=1, random_state=0)
    return CalibratedClassifierCV(base, method="isotonic", cv=N_FOLDS).fit(X, y)


@pytest.fixture(scope="module")
def explainer(tiny_model) -> ShapExplainer:
    return ShapExplainer([pair.estimator.get_booster() for pair in tiny_model.calibrated_classifiers_])


def make_inputs(*, start_x: float = 70.0, start_y: float = 25.0, height: PassHeight = PassHeight.HIGH,
                body_part: BodyPart = BodyPart.FOOT, play_pattern: PlayPattern = PlayPattern.REGULAR,
                under_pressure: bool = True) -> PassInputs:
    return PassInputs(start_x=start_x, start_y=start_y, height=height, body_part=body_part,
                      play_pattern=play_pattern, under_pressure=under_pressure)


def make_event(seq: int = 10, *, completed: bool = True, **inputs) -> PassEvent:
    return PassEvent(
        event_id=f"m1:{seq}", match_id="m1", seq=seq, t=600.0, period=1, team_id="brk", player_id="brk_p7",
        inputs=make_inputs(**inputs),
        details=PassDetails(end_x=95.0, end_y=30.0, length=25.5, ball_speed_ms=16.0, completed=completed),
    )


def make_moment(event: PassEvent) -> Moment:
    xp = 0.1 if event.details.completed else 0.95  # always an outlier
    m = build_moment(event, xp, "tiny-model")
    assert m is not None
    return m


def explained(explainer: ShapExplainer, **kw) -> Explanation:
    event = make_event(**kw)
    return explainer.explain(event, make_moment(event))


def shap_row_with_units(**unit_values: float) -> list[float]:
    """A 17-column contribution row whose units sum to the given values (spread over the group's columns)."""
    row = [0.0] * len(FEATURE_ORDER)
    for unit in UNITS:
        total = unit_values.get(unit.name, 0.0)
        for col in unit.columns:
            row[FEATURE_ORDER.index(col)] = total / len(unit.columns)
    return row


def additive(e: Explanation) -> bool:
    return abs(e.baseline + sum(d.shap for d in e.drivers) + e.rest - e.raw_score) <= ADDITIVITY_TOLERANCE


# --- boosters property on XPassScorer -----------------------------------------------------------

def test_scorer_boosters_property(tiny_model, tmp_path):
    feature_list = tmp_path / "feature_list.json"
    feature_list.write_text(json.dumps(FEATURE_ORDER), encoding="utf-8")
    joblib.dump(tiny_model, tmp_path / "tiny.joblib")
    boosters = XPassScorer(tmp_path / "tiny.joblib", feature_list).boosters
    assert isinstance(boosters, tuple) and len(boosters) == N_FOLDS
    assert all(list(b.feature_names) == FEATURE_ORDER for b in boosters)

    plain = tiny_model.calibrated_classifiers_[0].estimator  # an XGBClassifier without calibration
    joblib.dump(plain, tmp_path / "plain.joblib")
    with pytest.raises(ModelLoadError, match="not a CalibratedClassifierCV"):
        XPassScorer(tmp_path / "plain.joblib", feature_list).boosters


# --- units and labels (no model) ----------------------------------------------------------------

def all_labels() -> list[str]:
    return [*POSITION_LABELS.values(), *HEIGHT_LABELS.values(), *BODY_PART_LABELS.values(),
            *PLAY_PATTERN_LABELS.values(), *PRESSURE_LABELS.values()]


def test_units_cover_feature_order():
    assert UNIT_NAMES == ["position", "height", "body_part", "play_pattern", "under_pressure"]
    assert [u.topic for u in UNITS] == ["pitch position", "pass height", "body part", "how the move started",
                                        "pressure"]
    claimed = [c for u in UNITS for c in u.columns]
    assert sorted(claimed) == sorted(FEATURE_ORDER) and len(claimed) == len(set(claimed))
    units = {u.name: u for u in UNITS}
    assert units["position"].columns == ("start_x", "start_y", "angle_to_goal", "dist_to_goal")
    for name in GROUPS:
        assert units[name].columns == tuple(c for c in FEATURE_ORDER if c.startswith(f"{name}_"))


def test_feature_outside_every_unit_raises():
    with pytest.raises(ExplanationError, match="no explanation unit"):
        build_units([*FEATURE_ORDER, "pass_speed"])


def test_feature_in_two_units_raises():
    specs = [*UNIT_SPECS, ("again", "again", ("under_pressure",))]
    with pytest.raises(ExplanationError, match="more than one"):
        build_units(FEATURE_ORDER, specs)


@pytest.mark.parametrize("enum, labels", [(PassHeight, HEIGHT_LABELS), (BodyPart, BODY_PART_LABELS),
                                          (PlayPattern, PLAY_PATTERN_LABELS)])
def test_every_enum_member_has_a_label(enum, labels):
    missing = [m for m in enum if not labels.get(m)]
    assert not missing, f"{enum.__name__} members without a label: {missing}"


def test_labels_have_no_digits_or_banned_words():
    for label in all_labels():
        assert not any(ch.isdigit() for ch in label), label
        assert "not a" not in label and "expected" not in label, label
        if label != PLAY_PATTERN_LABELS[PlayPattern.THROW_IN]:  # "from a throw-in" names the restart
            assert "throw" not in label, label
    assert PRESSURE_LABELS == {True: "under pressure", False: "with no pressure on the passer"}
    assert BODY_PART_LABELS[BodyPart.KEEPER_ARM] == "from the goalkeeper's hands"
    assert not any(w in label for label in POSITION_LABELS.values() for w in ("left", "right"))


L, W = PITCH_LENGTH, PITCH_WIDTH
EPS = 1e-6


@pytest.mark.parametrize("fraction, below, at_or_above", [
    (DEPTH_MIDDLE_FROM, "defensive_third", "middle_third"),
    (DEPTH_ATTACKING_FROM, "middle_third", "attacking_third"),
    (DEPTH_NEAR_GOAL_FROM, "attacking_third", "near_goal"),
])
def test_depth_thresholds(fraction, below, at_or_above):
    x = fraction * L
    assert depth_zone(x - EPS) == below
    assert depth_zone(x) == at_or_above
    assert depth_zone(x + EPS) == at_or_above


@pytest.mark.parametrize("fraction, below, at_or_above", [
    (WIDTH_OFF_CENTRE_FROM, "central", "off_centre"),
    (WIDTH_OUT_WIDE_FROM, "off_centre", "out_wide"),
])
def test_width_thresholds(fraction, below, at_or_above):
    offset = fraction * W
    for y in (W / 2 + offset, W / 2 - offset):  # both sides of the middle
        step = EPS if y > W / 2 else -EPS
        assert width_zone(y - step) == below
        assert width_zone(y) == at_or_above
        assert width_zone(y + step) == at_or_above


def test_pitch_edges():
    assert depth_zone(0.0) == "defensive_third" and depth_zone(L) == "near_goal"
    assert width_zone(W / 2) == "central" and width_zone(0.0) == width_zone(W) == "out_wide"


def test_all_twelve_position_labels_reachable():
    xs = [0.1 * L, 0.5 * L, 0.75 * L, 0.95 * L]
    ys = [0.5 * W, 0.5 * W + 0.2 * W, 0.02 * W]
    seen = {position_value_and_label(x, y) for x in xs for y in ys}
    assert {label for _, label in seen} == set(POSITION_LABELS.values())
    assert {value for value, _ in seen} == {f"{d}|{w}" for d, w in POSITION_LABELS}


@pytest.mark.parametrize("y", [0.0, 3.3, 11.99, 12.0, 14.0, 25.5, 28.0, 39.9, 40.0])
def test_mirrored_y_gives_same_label(y):
    for x in (5.0, 60.0, 90.0, 110.0):
        assert position_value_and_label(x, y) == position_value_and_label(x, W - y)


def test_unit_shap_sums_unit_columns():
    row = list(np.arange(1, len(FEATURE_ORDER) + 1, dtype=float))
    sums = dict(zip(UNIT_NAMES, unit_shap(row)))
    for u in UNITS:
        assert sums[u.name] == sum(row[FEATURE_ORDER.index(c)] for c in u.columns)


def test_rank_units_ties_and_zero():
    shaps = [0.4, -0.4, 0.0, 0.9, 0.1]
    assert rank_units(shaps) == [3, 0, 1]  # |0.4| tie: position (0) before height (1)
    assert rank_units([0.0] * 5) == []
    assert rank_units([ZERO_SHAP / 2, 0.0, 0.3, 0.0, 0.0]) == [2]


def test_values_and_labels_come_from_inputs():
    inputs = make_inputs(start_x=110.0, start_y=4.0, height=PassHeight.GROUND, body_part=BodyPart.HEAD,
                         play_pattern=PlayPattern.THROW_IN, under_pressure=False)
    row = shap_row_with_units(position=0.6, height=0.5, body_part=-0.4, play_pattern=0.3, under_pressure=0.2)
    drivers, _ = drivers_and_rest(row, inputs)
    assert [(d.feature, d.topic, d.value, d.label, d.direction) for d in drivers] == [
        ("position", "pitch position", "near_goal|out_wide", "from close to the opposition goal, out wide", "raises"),
        ("height", "pass height", "Ground Pass", "along the ground", "raises"),
        ("body_part", "body part", "Head", "with a header", "lowers"),
    ]


def test_pressure_values():
    row = shap_row_with_units(under_pressure=-0.6)
    for pressed, label in ((True, "under pressure"), (False, "with no pressure on the passer")):
        (d,), _ = drivers_and_rest(row, make_inputs(under_pressure=pressed))
        assert (d.feature, d.value, d.label) == ("under_pressure", float(pressed), label)


def test_one_nonzero_unit_gives_one_driver():
    row = shap_row_with_units(body_part=0.7)
    drivers, rest = drivers_and_rest(row, make_inputs())
    assert [(d.rank, d.feature, d.direction) for d in drivers] == [(1, "body_part", "raises")]
    assert rest == pytest.approx(sum(row) - drivers[0].shap, abs=1e-12)


def test_all_zero_gives_no_drivers():
    row = [0.0] * len(FEATURE_ORDER)
    drivers, rest = drivers_and_rest(row, make_inputs())
    assert drivers == [] and rest == 0.0
    e = Explanation(moment_id="moment:x", event_id="x", method="m", unit="u", baseline=1.25, raw_score=1.25,
                    drivers=drivers, rest=rest, evidence={}, model_version="v")
    assert additive(e)


def test_rest_is_the_sum_of_other_units():
    row = shap_row_with_units(position=0.5, height=0.3, body_part=-0.2, play_pattern=0.05, under_pressure=-0.4)
    drivers, rest = drivers_and_rest(row, make_inputs())
    assert [d.feature for d in drivers] == ["position", "under_pressure", "height"]
    assert rest == pytest.approx(-0.2 + 0.05, abs=1e-12)


# --- ShapExplainer (tiny model) -----------------------------------------------------------------

def test_group_shap_matches_raw_pred_contribs(explainer, tiny_model):
    event = [e for e in generate_match(42)[0] if e.type == "pass"][0]
    e = explainer.explain(event, build_moment(event, 0.0 if event.details.completed else 1.0, "tiny-model"))
    contribs, _ = explainer.contributions([event.inputs])
    raw = contribs[0, :-1]
    sums = dict(zip(UNIT_NAMES, unit_shap(raw)))
    for d in e.drivers:
        unit = next(u for u in UNITS if u.name == d.feature)
        assert d.shap == pytest.approx(sum(raw[FEATURE_ORDER.index(c)] for c in unit.columns), abs=1e-12)
        assert d.shap == sums[d.feature]
    position_cols = ("start_x", "start_y", "angle_to_goal", "dist_to_goal")
    assert sums["position"] == pytest.approx(sum(raw[FEATURE_ORDER.index(c)] for c in position_cols), abs=1e-12)
    # the averaged contributions equal the mean of each booster's own pred_contribs
    dm = xgb.DMatrix(build_features([event.inputs]))
    per_fold = [p.estimator.get_booster().predict(dm, pred_contribs=True) for p in tiny_model.calibrated_classifiers_]
    assert np.allclose(contribs, np.mean(per_fold, axis=0), atol=1e-6)


def test_additivity_and_driver_shape(explainer):
    e = explained(explainer, body_part=BodyPart.HEAD, play_pattern=PlayPattern.THROW_IN)
    assert additive(e)
    assert 0 < len(e.drivers) <= 3
    assert [d.rank for d in e.drivers] == list(range(1, len(e.drivers) + 1))
    keys = [(-abs(d.shap), UNIT_NAMES.index(d.feature)) for d in e.drivers]
    assert keys == sorted(keys)
    for d in e.drivers:
        assert d.direction == ("raises" if d.shap > 0 else "lowers")
        assert d.feature in UNIT_NAMES
        if d.feature != "under_pressure":
            assert isinstance(d.value, str)
        assert not d.label.startswith("not a") and not any(ch.isdigit() for ch in d.label)


def test_raw_score_is_mean_of_fold_margins(explainer, tiny_model):
    event = make_event()
    e = explainer.explain(event, make_moment(event))
    X = build_features([event.inputs])
    margins = [pair.estimator.predict(X, output_margin=True)[0] for pair in tiny_model.calibrated_classifiers_]
    assert e.raw_score == pytest.approx(float(np.mean(margins)), abs=1e-6)
    assert e.method == f"xgboost pred_contribs, mean of {N_FOLDS} fold models"
    assert "not percentage points of xP" in e.unit


def test_deterministic_json(explainer):
    assert explained(explainer).model_dump_json() == explained(explainer).model_dump_json()


def test_evidence_keys_and_order(explainer):
    e = explained(explainer, seq=4, body_part=BodyPart.HEAD)
    fields = ("topic", "value", "label", "direction", "shap")
    expected = [f"m1:4:driver{d.rank}_{name}" for d in e.drivers for name in fields]
    assert list(e.evidence) == expected + ["m1:4:baseline", "m1:4:raw_score"]
    for d in e.drivers:
        for name in fields:
            assert e.evidence[f"m1:4:driver{d.rank}_{name}"] == getattr(d, name)
    assert e.evidence["m1:4:baseline"] == e.baseline and e.evidence["m1:4:raw_score"] == e.raw_score


def test_evidence_only_for_existing_drivers():
    drivers, rest = drivers_and_rest(shap_row_with_units(height=0.3), make_inputs())
    ev = build_explanation_evidence("x", drivers, 1.0, 1.3)
    assert list(ev) == ["x:driver1_topic", "x:driver1_value", "x:driver1_label", "x:driver1_direction",
                        "x:driver1_shap", "x:baseline", "x:raw_score"]


def test_json_round_trip_keeps_types(explainer):
    e = explained(explainer, body_part=BodyPart.HEAD)
    back = Explanation.model_validate_json(e.model_dump_json())
    assert back == e
    for d in back.drivers:
        assert type(d.value) is (float if d.feature == "under_pressure" else str)


def test_mismatched_ids_raise(explainer):
    with pytest.raises(ValueError, match="not m1:2"):
        explainer.explain(make_event(seq=2), make_moment(make_event(seq=1)))


def test_explain_many_matches_explain(explainer):
    events = [make_event(seq=i, start_x=10.0 + 20 * i, under_pressure=bool(i % 2),
                         height=(PassHeight.GROUND, PassHeight.HIGH, PassHeight.LOW)[i % 3]) for i in range(5)]
    pairs = [(ev, make_moment(ev)) for ev in events]
    assert explainer.explain_many(pairs) == [explainer.explain(ev, m) for ev, m in pairs]
    assert explainer.explain_many([]) == []


def test_outcome_does_not_change_explanation(explainer):
    done, failed = make_event(completed=True), make_event(completed=False)
    moment = make_moment(done)
    assert explainer.explain(done, moment) == explainer.explain(failed, moment)


def test_explanation_is_strict(explainer):
    e = explained(explainer)
    with pytest.raises(ValidationError):
        Explanation.model_validate({**e.model_dump(), "drivers": e.model_dump()["drivers"] * 2})
    with pytest.raises(ValidationError):
        Driver.model_validate({**e.drivers[0].model_dump(), "rank": 4})
    with pytest.raises(ValidationError):
        Driver.model_validate({**e.drivers[0].model_dump(), "direction": "neutral"})
    with pytest.raises(ValidationError):
        Explanation.model_validate({**e.model_dump(), "extra": 1})
    with pytest.raises(ValidationError):
        e.rest = 0.0


def test_booster_feature_mismatch_raises():
    rng = np.random.default_rng(0)
    X = random_features(50, rng).rename(columns={"start_x": "sx"})
    other = XGBClassifier(n_estimators=2, n_jobs=1).fit(X, rng.integers(0, 2, 50))
    with pytest.raises(ExplanationError, match="FEATURE_ORDER"):
        ShapExplainer([other.get_booster()])


# --- real model (skipped when the joblib file is missing) ---------------------------------------

def test_real_model_first_moment_additivity():
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc})")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path}")
    scorer = XPassScorer(model_path, feature_list_path)
    passes = [e for e in generate_match(42)[0] if e.type == "pass"]
    results = scorer.score([p.inputs for p in passes])
    event, moment = next(
        (p, m) for p, r in zip(passes, results)
        if (m := build_moment(p, r.expected_probability, scorer.model_version)) is not None
    )
    explainer = ShapExplainer(scorer.boosters)
    e = explainer.explain(event, moment)
    assert len(scorer.boosters) == 5
    assert e.method == "xgboost pred_contribs, mean of 5 fold models"
    assert additive(e)
    raw = explainer.contributions([event.inputs])[0][0, :-1]
    position_cols = ("start_x", "start_y", "angle_to_goal", "dist_to_goal")
    assert unit_shap(raw)[0] == pytest.approx(sum(raw[FEATURE_ORDER.index(c)] for c in position_cols), abs=1e-12)
    for d in e.drivers:
        unit = next(u for u in UNITS if u.name == d.feature)
        assert d.shap == pytest.approx(sum(raw[FEATURE_ORDER.index(c)] for c in unit.columns), abs=1e-12)
