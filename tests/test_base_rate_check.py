import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import base_rate_check as brc  # noqa: E402

from xplain.config import AnalyticsConfig  # noqa: E402


def test_base_rates_count_expected_and_observed_in_each_zone():
    cfg = AnalyticsConfig()   # 0.35 / 0.92, unchanged
    xps = [0.95, 0.93, 0.92, 0.50, 0.35, 0.10]
    completed = [True, False, True, False, True, False]
    r = brc.base_rates(xps, completed, cfg)
    assert r["passes"] == 6
    f, s = r["failure_zone"], r["success_zone"]
    assert (f["threshold"], f["passes"], f["observed_failures"]) == (0.92, 3, 1)
    assert f["expected_failures"] == pytest.approx(0.05 + 0.07 + 0.08)
    assert (s["threshold"], s["passes"], s["observed_successes"]) == (0.35, 2, 1)
    assert s["expected_successes"] == pytest.approx(0.45)


def test_empty_zones():
    r = brc.base_rates([0.5, 0.6], [True, False])
    assert r["failure_zone"]["passes"] == 0 and r["failure_zone"]["expected_failures"] == 0
    assert r["success_zone"]["passes"] == 0 and r["success_zone"]["observed_successes"] == 0


def test_parse_seeds():
    assert brc.parse_seeds("1001-1003") == [1001, 1002, 1003]
    assert brc.parse_seeds("7") == [7]
