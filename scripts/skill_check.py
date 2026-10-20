"""
Hidden-skill check: does PVA recover the generator's hidden pass skill? Offline: generated matches, the default
xPass model, no model calls.

The generator draws each player's pass skill fresh per match (generator._make_teams) and writes it only to the
truth dict, never to the events. Per player and match: PVA per pass = (completed - the xP sum) / passes, and the
raw completion rate. Over the players with at least --min-passes passes it prints:
  * Pearson (and Spearman) of PVA per pass with the hidden skill, and Pearson of the raw completion rate with it
  * how often the match's top player by PVA per pass is in the top 3 by hidden skill among those players, with
    the chance rate (3 / the players per match), and the median skill rank of that player
  * the mean PVA per pass by hidden-skill quartile

This validates on synthetic data that the pipeline separates player quality from pass difficulty (the generator
was designed with a skill term, config.GenConfig.w_skill). It is not a measure of real-world scouting accuracy.

Usage:
    python scripts/skill_check.py [--seeds 2001-2100] [--min-passes 20]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from xplain.generator import generate_match  # noqa: E402
from xplain.schema import PassEvent  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402

TOP = 3


@dataclass(frozen=True)
class PlayerMatch:
    match: int            # the seed
    player_id: str
    passes: int
    completed: int
    xp_sum: float
    skill: float          # the hidden pass skill (truth only)

    @property
    def pva_per_pass(self) -> float:
        return (self.completed - self.xp_sum) / self.passes

    @property
    def completion(self) -> float:
        return self.completed / self.passes


def player_matches(seed: int, passes: Sequence[PassEvent], xps: Sequence[float], truth: dict) -> list[PlayerMatch]:
    """One row per player with at least one pass in the match, in first-pass order."""
    skill = {p["player_id"]: p["pass_skill"] for p in truth["players"]}
    agg: dict[str, list] = {}
    for p, x in zip(passes, xps, strict=True):
        a = agg.setdefault(p.player_id, [0, 0, 0.0])
        a[0] += 1
        a[1] += p.details.completed
        a[2] += x
    return [PlayerMatch(seed, pid, n, c, xp, skill[pid]) for pid, (n, c, xp) in agg.items()]


def correlations(rows: Sequence[PlayerMatch]) -> dict:
    """PVA per pass and the raw completion rate against the hidden skill, pooled over player-matches."""
    skill = [r.skill for r in rows]
    pva = [r.pva_per_pass for r in rows]
    return {"n": len(rows),
            "pva_pearson": float(pearsonr(pva, skill)[0]),
            "pva_spearman": float(spearmanr(pva, skill)[0]),
            "completion_pearson": float(pearsonr([r.completion for r in rows], skill)[0])}


def top_pva_rank(rows: Sequence[PlayerMatch]) -> tuple[int, int]:
    """The hidden-skill rank (1 = most skilled) of the match's top player by PVA per pass, and the pool size."""
    top = max(rows, key=lambda r: r.pva_per_pass)
    order = sorted(rows, key=lambda r: -r.skill)
    return order.index(top) + 1, len(rows)


def top_hits(by_match: dict[int, list[PlayerMatch]], top: int = TOP) -> dict:
    """How often the top-PVA player is in the hidden top `top`, against chance (top / pool size per match)."""
    ranks = [top_pva_rank(rows) for rows in by_match.values() if rows]
    return {"matches": len(ranks),
            "hit_rate": sum(r <= top for r, _ in ranks) / len(ranks),
            "chance": statistics.mean(min(top, n) / n for _, n in ranks),
            "median_rank": float(statistics.median(r for r, _ in ranks)),
            "mean_pool": statistics.mean(n for _, n in ranks)}


def quartile_means(rows: Sequence[PlayerMatch]) -> list[float]:
    """Mean PVA per pass in each hidden-skill quartile, lowest skill first."""
    q1, q2, q3 = np.quantile([r.skill for r in rows], [0.25, 0.5, 0.75])
    edges = [(-np.inf, q1), (q1, q2), (q2, q3), (q3, np.inf)]
    return [statistics.mean(r.pva_per_pass for r in rows if lo <= r.skill < hi) for lo, hi in edges]


def parse_seeds(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def run(seeds: Sequence[int], min_passes: int, scorer: XPassScorer) -> dict:
    every: list[PlayerMatch] = []
    for seed in seeds:
        events, truth = generate_match(seed)
        passes = [e for e in events if isinstance(e, PassEvent)]
        xps = [r.expected_probability for r in scorer.score([p.inputs for p in passes])]
        every += player_matches(seed, passes, xps, truth)
    kept = [r for r in every if r.passes >= min_passes]
    by_match: dict[int, list[PlayerMatch]] = {}
    for r in kept:
        by_match.setdefault(r.match, []).append(r)
    return {"model": scorer.model_version, "seeds": list(seeds), "min_passes": min_passes,
            "player_matches": len(every), "median_passes": float(statistics.median(r.passes for r in every)),
            **correlations(kept), "top": top_hits(by_match), "quartiles": quartile_means(kept)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Does PVA per pass recover the generator's hidden pass skill?")
    parser.add_argument("--seeds", default="2001-2100", help="seed range (default: the 100 hold-out matches)")
    parser.add_argument("--min-passes", type=int, default=20, help="players with at least this many passes")
    args = parser.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    r = run(seeds, args.min_passes, XPassScorer(*find_default_model()))
    t = r["top"]
    print(f"model {r['model']}, seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} matches), {r['player_matches']} "
          f"player-matches, median {r['median_passes']:g} passes")
    print(f"players with >= {r['min_passes']} passes: {r['n']}")
    print(f"PVA per pass vs hidden skill: Pearson {r['pva_pearson']:.2f} (Spearman {r['pva_spearman']:.2f}); "
          f"raw completion rate: Pearson {r['completion_pearson']:.2f}")
    print(f"top-PVA player in the hidden top {TOP}: {t['hit_rate']:.0%} of {t['matches']} matches "
          f"(chance {t['chance']:.0%}); median skill rank {t['median_rank']:g} of ~{t['mean_pool']:.0f}")
    print("PVA per pass by skill quartile (low to high): " + " / ".join(f"{q:+.3f}" for q in r["quartiles"]))
    print("Synthetic data: this checks the pipeline against the generator's own skill term, "
          "not real-world scouting accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
