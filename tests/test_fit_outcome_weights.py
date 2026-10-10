import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_outcome_weights.py"
_spec = importlib.util.spec_from_file_location("fit_outcome_weights", _PATH)
fow = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fow  # dataclasses look the module up while the class is created
_spec.loader.exec_module(fow)


def _event(poss, start, end, *, ptype=None, height="Ground Pass", outcome=None, pressure=False,
           pattern="Regular Play", position="Center Back", etype="Pass"):
    p = {"end_location": end, "height": {"name": height}, "length": 999.0}
    if ptype:
        p["type"] = {"name": ptype}
    if outcome:
        p["outcome"] = {"name": outcome}
    e = {"type": {"name": etype}, "possession": poss, "play_pattern": {"name": pattern},
         "position": {"name": position}, "team": {"id": 1}, "possession_team": {"id": 1},
         "location": start, "pass": p}
    if pressure:
        e["under_pressure"] = True
    return e


def test_pass_features_geometry_and_flags():
    row = fow.pass_features(_event(1, [30.0, 40.0], [60.0, 80.0], height="High Pass", pressure=True))
    assert row["length"] == pytest.approx(50.0)          # recomputed, not StatsBomb's length field
    assert row["forward_gain"] == pytest.approx(30.0)
    assert (row["under_pressure"], row["is_high"], row["is_low"], row["completed"]) == (1.0, 1.0, 0.0, 1.0)

    back = fow.pass_features(_event(1, [60.0, 40.0], [50.0, 40.0], height="Low Pass", outcome="Out"))
    assert back["forward_gain"] == 0.0
    assert (back["is_high"], back["is_low"], back["completed"]) == (0.0, 1.0, 0.0)


@pytest.mark.parametrize("ptype,key", [("Kick Off", "kick_off"), ("Throw-in", "throw_in"),
                                       ("Goal Kick", "goal_kick"), ("Free Kick", "free_kick"),
                                       ("Corner", "corner")])
def test_restart_flags_from_pass_type(ptype, key):
    row = fow.pass_features(_event(1, [0.0, 0.0], [3.0, 4.0], ptype=ptype))
    assert row[key] == 1.0
    assert sum(row[r] for r in fow.RESTARTS) == 1.0


def test_match_rows_drops_and_keeper_flag():
    events = [
        {"type": {"name": "Shot"}, "possession": 1},
        _event(2, [5.0, 40.0], [40.0, 30.0], pattern="From Keeper", position="Goalkeeper"),   # keeper
        _event(2, [40.0, 30.0], [50.0, 30.0], pattern="From Keeper", position="Goalkeeper"),  # not first
        _event(3, [5.0, 40.0], [40.0, 30.0], pattern="From Keeper", position="Left Back"),    # not GK
        _event(4, [10.0, 10.0], None),                                                       # no end
    ]
    rows = list(fow.match_rows(events))
    assert len(rows) == 3
    assert [r["keeper"] for r in rows] == [1.0, 0.0, 0.0]
    assert math.isclose(rows[0]["length"], math.hypot(35.0, 10.0))


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 60_000
    length = rng.gamma(2.0, 9.0, n)
    pressure = (rng.random(n) < 0.2).astype(float)
    X = np.column_stack([length, pressure])
    true_b0, true_b = 2.5, np.array([-0.05, -0.6])
    y = (rng.random(n) < 1 / (1 + np.exp(-(true_b0 + X @ true_b)))).astype(float)

    b0, b = fow.fit_logit(X, y)
    assert b0 == pytest.approx(true_b0, abs=0.1)
    assert b == pytest.approx(true_b, abs=0.05)
    assert fow.predict_proba(b0, b, X).mean() == pytest.approx(y.mean(), abs=1e-4)


def test_location_features_match_features_module():
    from xplain.features import angle_to_goal, dist_to_goal

    row = fow.pass_features(_event(1, [30.0, 10.0], [40.0, 10.0]))
    assert row["start_x"] == 30.0
    assert row["dist_to_goal"] == pytest.approx(float(dist_to_goal(30.0, 10.0)))
    assert row["angle_to_goal"] == pytest.approx(float(angle_to_goal(30.0, 10.0)))


def test_bucket_edges_and_rates():
    cols = {
        "is_high": np.array([0, 1, 0, 0.0]),
        "is_low": np.array([0, 0, 1, 0.0]),
        "length": np.array([9.99, 10.0, 45.0, 30.0]),
        "start_x": np.array([0.0, 30.0, 89.9, 120.0]),
        "under_pressure": np.array([0, 1, 0, 1.0]),
        **{r: np.zeros(4) for r in fow.RESTARTS},
    }
    cols["corner"][3] = 1.0
    labels = fow.assign_buckets(cols)
    assert list(labels["height"]) == ["ground", "high", "low", "ground"]
    assert list(labels["length"]) == ["0-10", "10-20", "45+", "30-45"]
    assert list(labels["start_x"]) == ["0-30", "30-60", "60-90", "90-120"]
    assert list(labels["restart"]) == ["none", "none", "none", "corner"]
    rates = fow.bucket_rates(labels, np.array([1, 0, 1, 0.0]))
    assert rates["under_pressure"]["0"] == {"n": 2, "rate": 1.0}
    assert rates["restart"]["kick_off"] == {"n": 0, "rate": None}


def _synthetic_logit_data(true_b0, true_b, n=60_000, seed=1):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.gamma(2.0, 9.0, n), (rng.random(n) < 0.2).astype(float), rng.uniform(0, 120, n)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(true_b0 + X @ np.asarray(true_b))))).astype(float)
    return X, y


def test_bounded_fit_without_active_bound_matches_unconstrained():
    # true coefficients already satisfy the bounds, so the constrained optimum is the unconstrained one
    X, y = _synthetic_logit_data(2.5, [-0.05, -0.6, 0.01])
    b0, b = fow.fit_logit(X, y)
    c0, c = fow.fit_logit_bounded(X, y, [True, True, False])
    assert c0 == pytest.approx(b0, abs=1e-3)
    assert c == pytest.approx(b, abs=1e-4)


def test_bounded_fit_pins_wrong_signed_coefficient_at_zero():
    X, y = _synthetic_logit_data(1.0, [0.03, -0.6, 0.01])
    _, c = fow.fit_logit_bounded(X, y, [True, True, False])
    assert c[0] == pytest.approx(0.0, abs=1e-9)
    assert c[1] < 0


def test_variant_d_is_start_only():
    assert "length" not in fow.FEATURES_D and "forward_gain" not in fow.FEATURES_D
    assert {"start_x", "dist_to_goal", "angle_to_goal", *fow.RESTARTS} <= set(fow.FEATURES_D)
    assert set(fow.NONPOSITIVE_D) == {"under_pressure", "is_high", "is_low"}
