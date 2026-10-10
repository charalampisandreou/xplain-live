import socket
from collections import Counter

import pytest

from xplain import replay


def test_mmss():
    assert replay.mmss(0.0) == "00:00"
    assert replay.mmss(75.9) == "01:15"
    assert replay.mmss(5807.51) == "96:47"


def test_p95():
    assert replay.p95([float(i) for i in range(1, 101)]) == 95.0
    assert replay.p95([3.0]) == 3.0


def test_summary_counts_and_top_flagged():
    rows = [
        {"seq": 1, "t": 10.0, "xp": 0.9, "difficulty": 10, "difficulty_band": "routine", "completed": True,
         "delta": 0.1, "outlier": None, "player_id": "a-p1", "model_version": "m"},
        {"seq": 2, "t": 20.0, "xp": 0.3, "difficulty": 70, "difficulty_band": "exceptional", "completed": True,
         "delta": 0.7, "outlier": "surprise_success", "player_id": "a-p2", "model_version": "m"},
        {"seq": 3, "t": 30.0, "xp": 0.95, "difficulty": 5, "difficulty_band": "routine", "completed": False,
         "delta": -0.95, "outlier": "surprise_failure", "player_id": "b-p3", "model_version": "m"},
    ]
    s = replay.summarize(rows, [5.0, 7.0, 9.0], Counter({"pass": 3, "shot": 1}))
    assert s["passes_scored"] == 3
    assert s["bands"] == {"routine": 2, "moderate": 0, "hard": 0, "exceptional": 1}
    assert s["outliers"] == {"surprise_success": 1, "surprise_failure": 1}
    assert [r["seq"] for r in s["top_flagged"]] == [3, 2]
    assert s["latency_median_ms"] == 7.0


def test_unreachable_server_gives_clear_error(tmp_path):
    with socket.socket() as sock:  # grab a free port, then close it so nothing listens there
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    match = tmp_path / "m.jsonl"
    match.write_text("", encoding="utf-8")
    with pytest.raises(replay.ServerUnreachable, match="Cannot reach the scoring API"):
        replay.replay(match, f"http://127.0.0.1:{port}", speed=0)
    assert replay.main(["--file", str(match), "--url", f"http://127.0.0.1:{port}"]) == 2
