"""
Feature builder for the xPass model.

Turns ``PassInputs`` into the exact 17 columns, in the exact order, the trained model expects.
The one-hot baselines match training: ground pass (height); any body part or play pattern
without its own column, including "Other" (body part and play pattern).
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import BodyPart, PassHeight, PassInputs, PlayPattern

GOAL_X = 120.0
GOAL_Y = 40.0

# Must equal FEAT_TRUEBEST in xPass-Project (commit bc29660) and models/feature_list.json.
FEATURE_ORDER: list[str] = [
    "start_x",
    "start_y",
    "angle_to_goal",
    "dist_to_goal",
    "height_Low Pass",
    "height_High Pass",
    "body_part_Foot",
    "body_part_Head",
    "body_part_Keeper Arm",
    "play_pattern_Regular Play",
    "play_pattern_Kick Off",
    "play_pattern_Throw In",
    "play_pattern_Free Kick",
    "play_pattern_Goal Kick",
    "play_pattern_Keeper",
    "play_pattern_Corner",
    "under_pressure",
]


class FeatureOrderError(ValueError):
    """The model's feature_list.json does not match the feature builder."""


def angle_to_goal(start_x, start_y):
    """Angle (radians) from the pass start to the goal centre. Same formula as training."""
    return np.arctan2(GOAL_Y - start_y, GOAL_X - start_x)


def dist_to_goal(start_x, start_y):
    return np.sqrt((GOAL_X - start_x) ** 2 + (GOAL_Y - start_y) ** 2)


def validate_feature_order(feature_list_path: str | Path) -> list[str]:
    """Raise FeatureOrderError unless the saved list equals FEATURE_ORDER, same order."""
    saved = json.loads(Path(feature_list_path).read_text(encoding="utf-8"))
    if saved != FEATURE_ORDER:
        missing = [c for c in FEATURE_ORDER if c not in saved]
        extra = [c for c in saved if c not in FEATURE_ORDER]
        detail = f"missing={missing}, extra={extra}" if (missing or extra) else "same columns, different order"
        raise FeatureOrderError(f"{feature_list_path} does not match FEATURE_ORDER ({detail}).")
    return saved


def build_features(passes: Sequence[PassInputs]) -> pd.DataFrame:
    """Build the model matrix. Accepts PassInputs only, so end-derived fields cannot leak in."""
    for p in passes:
        if not isinstance(p, PassInputs):
            raise TypeError(f"build_features takes PassInputs, got {type(p).__name__}")

    sx = np.array([p.start_x for p in passes], dtype=float)
    sy = np.array([p.start_y for p in passes], dtype=float)

    cols = {
        "start_x": sx,
        "start_y": sy,
        "angle_to_goal": angle_to_goal(sx, sy),
        "dist_to_goal": dist_to_goal(sx, sy),
        "height_Low Pass": [int(p.height == PassHeight.LOW) for p in passes],
        "height_High Pass": [int(p.height == PassHeight.HIGH) for p in passes],
        "body_part_Foot": [int(p.body_part == BodyPart.FOOT) for p in passes],
        "body_part_Head": [int(p.body_part == BodyPart.HEAD) for p in passes],
        "body_part_Keeper Arm": [int(p.body_part == BodyPart.KEEPER_ARM) for p in passes],
        "play_pattern_Regular Play": [int(p.play_pattern == PlayPattern.REGULAR) for p in passes],
        "play_pattern_Kick Off": [int(p.play_pattern == PlayPattern.KICK_OFF) for p in passes],
        "play_pattern_Throw In": [int(p.play_pattern == PlayPattern.THROW_IN) for p in passes],
        "play_pattern_Free Kick": [int(p.play_pattern == PlayPattern.FREE_KICK) for p in passes],
        "play_pattern_Goal Kick": [int(p.play_pattern == PlayPattern.GOAL_KICK) for p in passes],
        "play_pattern_Keeper": [int(p.play_pattern == PlayPattern.KEEPER) for p in passes],
        "play_pattern_Corner": [int(p.play_pattern == PlayPattern.CORNER) for p in passes],
        "under_pressure": [int(p.under_pressure) for p in passes],
    }
    return pd.DataFrame(cols, columns=FEATURE_ORDER)
