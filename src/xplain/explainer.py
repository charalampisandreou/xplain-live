"""
Explainer: up to three SHAP drivers behind the model's score for a Moment, one per pass input.

SHAP values come from XGBoost's own ``pred_contribs`` output for each calibration fold's booster,
averaged over the folds. They explain the raw (uncalibrated) model score in log-odds, not the
calibrated xP: describe them as drivers of the raw score, never as percentage points of xP.

The 17 model columns are summed into 5 units, one per pass input: pitch position (start_x,
start_y, angle_to_goal and dist_to_goal together, since all four describe where the pass starts),
the height, body part and play pattern one-hot groups, and under_pressure. A group's shap is the
sum of its columns, which also covers the group's omitted baseline category (ground pass, other
body part, other play pattern). Values and labels come from the event's PassInputs, never from
the 0/1 columns. Labels are phrases a fan can follow and contain no numbers; the numbers are in
the Moment evidence. Only ``event.inputs`` is used; the outcome never reaches the model.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xgboost as xgb

from .contracts import Driver, Explanation, Moment
from .features import FEATURE_ORDER, build_features
from .schema import PITCH_LENGTH, PITCH_WIDTH, BodyPart, PassEvent, PassHeight, PassInputs, PlayPattern

N_DRIVERS = 3
ZERO_SHAP = 1e-9                # a unit with |shap| below this is not reported as a driver
ADDITIVITY_TOLERANCE = 1e-4
UNIT = "log-odds of the raw model score, not percentage points of xP"


class ExplanationError(RuntimeError):
    """The SHAP contributions do not add up to the model's raw score, or the features do not fit."""


@dataclass(frozen=True)
class Unit:
    """One pass input: a name, a plain-English topic and the model columns it sums over."""

    name: str
    topic: str
    columns: tuple[str, ...]


# (name, topic, columns) in tie-break order. columns is either a tuple of exact column names or a
# prefix string meaning every FEATURE_ORDER column that starts with it.
UNIT_SPECS: tuple[tuple[str, str, tuple[str, ...] | str], ...] = (
    ("position", "pitch position", ("start_x", "start_y", "angle_to_goal", "dist_to_goal")),
    ("height", "pass height", "height_"),
    ("body_part", "body part", "body_part_"),
    ("play_pattern", "how the move started", "play_pattern_"),
    ("under_pressure", "pressure", ("under_pressure",)),
)

# Pitch zones as fractions of the pitch, so they scale with PITCH_LENGTH and PITCH_WIDTH. The
# passing team attacks towards high x. Display conventions (provisional), not measured values.
DEPTH_MIDDLE_FROM = 1 / 3       # start_x / length at or above this: middle third
DEPTH_ATTACKING_FROM = 2 / 3    # attacking third
DEPTH_NEAR_GOAL_FROM = 0.85     # close to the opposition goal
WIDTH_OFF_CENTRE_FROM = 0.15    # |start_y - width / 2| / width at or above this: off-centre
WIDTH_OUT_WIDE_FROM = 0.325     # out wide

POSITION_LABELS: dict[tuple[str, str], str] = {
    ("defensive_third", "central"): "from their own defensive third, centrally",
    ("defensive_third", "off_centre"): "from their own defensive third, off-centre",
    ("defensive_third", "out_wide"): "from their own defensive third, out wide",
    ("middle_third", "central"): "from the middle third, centrally",
    ("middle_third", "off_centre"): "from the middle third, off-centre",
    ("middle_third", "out_wide"): "from the middle third, out wide",
    ("attacking_third", "central"): "in the attacking third, centrally",
    ("attacking_third", "off_centre"): "in the attacking third, off-centre",
    ("attacking_third", "out_wide"): "in the attacking third, out wide",
    ("near_goal", "central"): "from close to the opposition goal, centrally",
    ("near_goal", "off_centre"): "from close to the opposition goal, off-centre",
    ("near_goal", "out_wide"): "from close to the opposition goal, out wide",
}
HEIGHT_LABELS: dict[PassHeight, str] = {
    PassHeight.GROUND: "along the ground",
    PassHeight.LOW: "played low",
    PassHeight.HIGH: "lofted high",
}
BODY_PART_LABELS: dict[BodyPart, str] = {
    BodyPart.FOOT: "with the foot",
    BodyPart.HEAD: "with a header",
    BodyPart.KEEPER_ARM: "from the goalkeeper's hands",
    BodyPart.OTHER: "with another part of the body",
}
PLAY_PATTERN_LABELS: dict[PlayPattern, str] = {
    PlayPattern.REGULAR: "in open play",
    PlayPattern.KICK_OFF: "from the kick-off",
    PlayPattern.THROW_IN: "from a throw-in",
    PlayPattern.FREE_KICK: "from a free kick",
    PlayPattern.GOAL_KICK: "from a goal kick",
    PlayPattern.KEEPER: "in a move started by the goalkeeper",
    PlayPattern.CORNER: "from a corner",
    PlayPattern.OTHER: "from another kind of play",
}
PRESSURE_LABELS: dict[bool, str] = {True: "under pressure", False: "with no pressure on the passer"}


def build_units(feature_order: Sequence[str] = FEATURE_ORDER,
                specs: Sequence[tuple[str, str, tuple[str, ...] | str]] = UNIT_SPECS) -> tuple[Unit, ...]:
    """Map the model columns onto the unit specs; raise ExplanationError if a column fits no unit or two."""
    units = []
    for name, topic, match in specs:
        if isinstance(match, str):
            cols = tuple(c for c in feature_order if c.startswith(match))
        else:
            absent = [c for c in match if c not in feature_order]
            if absent:
                raise ExplanationError(f"Unit {name!r} names columns not in the feature order: {absent}")
            cols = tuple(match)
        if not cols:
            raise ExplanationError(f"No model column for unit {name!r}")
        units.append(Unit(name, topic, cols))
    claimed = [c for u in units for c in u.columns]
    unclaimed = [c for c in feature_order if c not in claimed]
    if unclaimed:
        raise ExplanationError(f"Features in no explanation unit: {unclaimed}")
    doubled = sorted({c for c in claimed if claimed.count(c) > 1})
    if doubled:
        raise ExplanationError(f"Features in more than one explanation unit: {doubled}")
    return tuple(units)


UNITS: tuple[Unit, ...] = build_units()
_UNIT_INDEXES: tuple[tuple[int, ...], ...] = tuple(tuple(FEATURE_ORDER.index(c) for c in u.columns) for u in UNITS)


def unit_shap(shap_row: Sequence[float]) -> list[float]:
    """Sum the 17 column contributions (bias excluded) into one value per unit, in UNITS order."""
    if len(shap_row) != len(FEATURE_ORDER):
        raise ExplanationError(f"Expected {len(FEATURE_ORDER)} contributions, got {len(shap_row)}")
    return [float(sum(shap_row[i] for i in idx)) for idx in _UNIT_INDEXES]


def depth_zone(start_x: float) -> str:
    """Which third (or the area close to goal) the pass starts in, from start_x as a share of the length."""
    if start_x < DEPTH_MIDDLE_FROM * PITCH_LENGTH:
        return "defensive_third"
    if start_x < DEPTH_ATTACKING_FROM * PITCH_LENGTH:
        return "middle_third"
    if start_x < DEPTH_NEAR_GOAL_FROM * PITCH_LENGTH:
        return "attacking_third"
    return "near_goal"


def width_zone(start_y: float) -> str:
    """How far from the middle of the pitch the pass starts, from |start_y - width/2| as a share of the width."""
    off_centre = abs(start_y - PITCH_WIDTH / 2)
    if off_centre < WIDTH_OFF_CENTRE_FROM * PITCH_WIDTH:
        return "central"
    if off_centre < WIDTH_OUT_WIDE_FROM * PITCH_WIDTH:
        return "off_centre"
    return "out_wide"


def position_value_and_label(start_x: float, start_y: float) -> tuple[str, str]:
    """Position key such as "attacking_third|out_wide" and its label from POSITION_LABELS."""
    zone = (depth_zone(start_x), width_zone(start_y))
    return f"{zone[0]}|{zone[1]}", POSITION_LABELS[zone]


def unit_value_and_label(unit: Unit, inputs: PassInputs) -> tuple[float | str, str]:
    """What the pass had for this unit and the phrase for it, read from PassInputs."""
    if unit.name == "position":
        return position_value_and_label(inputs.start_x, inputs.start_y)
    if unit.name == "height":
        return inputs.height.value, HEIGHT_LABELS[inputs.height]
    if unit.name == "body_part":
        return inputs.body_part.value, BODY_PART_LABELS[inputs.body_part]
    if unit.name == "play_pattern":
        return inputs.play_pattern.value, PLAY_PATTERN_LABELS[inputs.play_pattern]
    if unit.name == "under_pressure":
        return float(inputs.under_pressure), PRESSURE_LABELS[inputs.under_pressure]
    raise ExplanationError(f"No label rule for unit {unit.name!r}")


def rank_units(shaps: Sequence[float], k: int = N_DRIVERS) -> list[int]:
    """Indices of the k largest |shap| units, largest first, then drop those below ZERO_SHAP; ties keep UNITS order."""
    top = sorted(range(len(shaps)), key=lambda i: (-abs(shaps[i]), i))[:k]
    return [i for i in top if abs(shaps[i]) >= ZERO_SHAP]


def drivers_and_rest(shap_row: Sequence[float], inputs: PassInputs) -> tuple[list[Driver], float]:
    """From one row of 17 column contributions and the pass inputs: 0 to 3 drivers, and the shap of all other units. No model needed."""
    shaps = unit_shap(shap_row)
    chosen = rank_units(shaps)
    drivers = []
    for rank, i in enumerate(chosen, start=1):
        value, label = unit_value_and_label(UNITS[i], inputs)
        drivers.append(Driver(
            rank=rank,
            feature=UNITS[i].name,
            topic=UNITS[i].topic,
            value=value,
            label=label,
            shap=shaps[i],
            direction="raises" if shaps[i] > 0 else "lowers",
        ))
    rest = float(sum(s for i, s in enumerate(shaps) if i not in chosen))
    return drivers, rest


def build_explanation_evidence(event_id: str, drivers: Sequence[Driver], baseline: float,
                               raw_score: float) -> dict[str, bool | int | float | str]:
    """Evidence keyed like the Analyst's, in a fixed order: per driver topic, value, label, direction, shap; then baseline and raw_score."""
    evidence: dict[str, bool | int | float | str] = {}
    for d in drivers:
        evidence[f"{event_id}:driver{d.rank}_topic"] = d.topic
        evidence[f"{event_id}:driver{d.rank}_value"] = d.value
        evidence[f"{event_id}:driver{d.rank}_label"] = d.label
        evidence[f"{event_id}:driver{d.rank}_direction"] = d.direction
        evidence[f"{event_id}:driver{d.rank}_shap"] = d.shap
    evidence[f"{event_id}:baseline"] = baseline
    evidence[f"{event_id}:raw_score"] = raw_score
    return evidence


class ShapExplainer:
    """Explains Moments with the averaged SHAP contributions of the calibration fold boosters."""

    def __init__(self, boosters: Sequence[xgb.Booster]):
        if not boosters:
            raise ExplanationError("ShapExplainer needs at least one booster")
        for b in boosters:
            if list(b.feature_names or []) != FEATURE_ORDER:
                raise ExplanationError(f"Booster feature names {b.feature_names} do not match FEATURE_ORDER")
        self._boosters = tuple(boosters)
        self.method = f"xgboost pred_contribs, mean of {len(self._boosters)} fold models"

    def contributions(self, inputs: Sequence[PassInputs]) -> tuple[np.ndarray, np.ndarray]:
        """Fold-averaged pred_contribs, shape (n, 18) with the bias last, and fold-averaged raw margins, shape (n,)."""
        dmatrix = xgb.DMatrix(build_features(inputs))  # inputs only, never details
        contribs = np.mean([b.predict(dmatrix, pred_contribs=True).astype(np.float64) for b in self._boosters], axis=0)
        margins = np.mean([b.predict(dmatrix, output_margin=True).astype(np.float64) for b in self._boosters], axis=0)
        return contribs, margins

    def explain(self, event: PassEvent, moment: Moment) -> Explanation:
        """Explain one Moment from its pass event."""
        return self.explain_many([(event, moment)])[0]

    def explain_many(self, pairs: Sequence[tuple[PassEvent, Moment]]) -> list[Explanation]:
        """Explain several Moments with one model call per booster."""
        if not pairs:
            return []
        for event, moment in pairs:
            if moment.event_id != event.event_id:
                raise ValueError(f"Moment {moment.moment_id} is for event {moment.event_id}, not {event.event_id}")

        contribs, margins = self.contributions([event.inputs for event, _ in pairs])
        explanations = []
        for row, (event, moment) in enumerate(pairs):
            shap_row, baseline, raw_score = contribs[row, :-1], float(contribs[row, -1]), float(margins[row])
            error = abs(baseline + float(shap_row.sum()) - raw_score)
            if error > ADDITIVITY_TOLERANCE:
                raise ExplanationError(
                    f"SHAP contributions for {event.event_id} miss the raw score by {error:.2e} "
                    f"(tolerance {ADDITIVITY_TOLERANCE:.0e})"
                )
            drivers, rest = drivers_and_rest(shap_row, event.inputs)
            explanations.append(Explanation(
                moment_id=moment.moment_id,
                event_id=event.event_id,
                method=self.method,
                unit=UNIT,
                baseline=baseline,
                raw_score=raw_score,
                drivers=drivers,
                rest=rest,
                evidence=build_explanation_evidence(event.event_id, drivers, baseline, raw_score),
                model_version=moment.model_version,
            ))
        return explanations
