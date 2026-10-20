import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import skill_check as sk  # noqa: E402

from xplain.generator import generate_match  # noqa: E402
from xplain.schema import PassEvent  # noqa: E402


def row(match, pid, passes, completed, xp_sum, skill):
    return sk.PlayerMatch(match, pid, passes, completed, xp_sum, skill)


def test_pva_per_pass_and_completion():
    r = row(1, "a", 40, 34, 30.0, 0.5)
    assert r.pva_per_pass == pytest.approx(0.1)
    assert r.completion == pytest.approx(0.85)


def test_player_matches_sum_per_player_and_take_the_hidden_skill():
    events, truth = generate_match(7)
    passes = [e for e in events if isinstance(e, PassEvent)]
    rows = sk.player_matches(7, passes, [0.5] * len(passes), truth)
    assert sum(r.passes for r in rows) == len(passes)
    assert sum(r.completed for r in rows) == sum(p.details.completed for p in passes)
    skill = {p["player_id"]: p["pass_skill"] for p in truth["players"]}
    assert all(r.skill == skill[r.player_id] and r.xp_sum == pytest.approx(0.5 * r.passes) for r in rows)


def test_top_pva_rank_ranks_by_hidden_skill():
    rows = [row(1, "a", 30, 20, 20.0, 2.0), row(1, "b", 30, 29, 20.0, 1.0), row(1, "c", 30, 10, 20.0, 0.0)]
    assert sk.top_pva_rank(rows) == (2, 3)   # b has the best PVA per pass and the second skill


def test_top_hits_against_chance():
    hit = [row(1, p, 30, c, 20.0, s) for p, c, s in [("a", 29, 3.0), ("b", 20, 2.0), ("c", 20, 1.0), ("d", 20, 0.0)]]
    miss = [row(2, p, 30, c, 20.0, s) for p, c, s in [("a", 20, 3.0), ("b", 20, 2.0), ("c", 20, 1.0), ("d", 29, 0.0)]]
    t = sk.top_hits({1: hit, 2: miss})
    assert t["matches"] == 2 and t["hit_rate"] == 0.5
    assert t["chance"] == pytest.approx(0.75) and t["median_rank"] == 2.5 and t["mean_pool"] == 4


def test_correlations_and_quartiles_follow_skill():
    rows = [row(1, str(i), 40, 20 + i, 25.0, float(i)) for i in range(8)]
    c = sk.correlations(rows)
    assert c["n"] == 8
    assert c["pva_pearson"] == pytest.approx(1.0) and c["pva_spearman"] == pytest.approx(1.0)
    assert c["completion_pearson"] == pytest.approx(1.0)
    q = sk.quartile_means(rows)
    assert len(q) == 4 and q == sorted(q)


def test_run_keeps_players_with_enough_passes():
    class Flat:   # every pass xP 0.8; no model file needed
        model_version = "flat"

        def score(self, inputs):
            return [SimpleNamespace(expected_probability=0.8) for _ in inputs]

    r = sk.run([7, 8], 20, Flat())
    assert r["model"] == "flat" and r["seeds"] == [7, 8] and r["top"]["matches"] == 2
    assert r["n"] <= r["player_matches"] and r["median_passes"] > 0
    assert -1 <= r["pva_pearson"] <= 1 and len(r["quartiles"]) == 4


def test_parse_seeds():
    assert sk.parse_seeds("2001-2003") == [2001, 2002, 2003]
    assert sk.parse_seeds("7") == [7]
