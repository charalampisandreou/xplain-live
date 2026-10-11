"""
Preview the Analyst's Moments for one generated match. Offline: no API server needed.

Loads the xPass model (XPassScorer), reads the match file (generating it with seed 42 when it is
missing), scores every pass in one batch, builds Moments and explains each one with SHAP. Prints
the count per kind, how many moments were explained and the largest additivity error, and the
Moment and Explanation JSON of the strongest moment of each kind.

Usage:
    python scripts/preview_moments.py [--file data/match_seed42.jsonl]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from xplain import generator  # noqa: E402
from xplain.analyst import OUTLIER_KINDS, build_moment, select_highlights  # noqa: E402
from xplain.contracts import Explanation, Moment  # noqa: E402
from xplain.explainer import ShapExplainer  # noqa: E402
from xplain.schema import Event, MatchMetaEvent, PassEvent  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "match_seed42.jsonl"
_EVENT = TypeAdapter(Event)


def load_events(path: Path) -> list:
    """Read a match JSONL file into schema events."""
    return [_EVENT.validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def names_from_meta(events: list) -> tuple[dict[str, str], dict[str, str]]:
    """Player and team names from the match_meta event, as (player_id -> name, team_id -> name)."""
    meta = next(e for e in events if isinstance(e, MatchMetaEvent))
    players = {p.player_id: p.name for p in meta.players}
    teams = {meta.home_team_id: meta.home_name, meta.away_team_id: meta.away_name}
    return players, teams


def build_all_moments(events: list, scorer: XPassScorer) -> tuple[int, list[Moment]]:
    """Score every pass in one batch and return (number of passes, moments)."""
    passes = [e for e in events if isinstance(e, PassEvent)]
    results = scorer.score([p.inputs for p in passes])
    players, teams = names_from_meta(events)
    moments = [
        m for p, r in zip(passes, results, strict=True)
        if (m := build_moment(p, r.expected_probability, scorer.model_version,
                              player_names=players, team_names=teams)) is not None
    ]
    return len(passes), moments


def explain_all(events: list, moments: list[Moment], scorer: XPassScorer) -> dict[str, Explanation]:
    """Explain every moment in one batch; returns explanations keyed by moment_id."""
    by_id = {e.event_id: e for e in events if isinstance(e, PassEvent)}
    explanations = ShapExplainer(scorer.boosters).explain_many([(by_id[m.event_id], m) for m in moments])
    return {x.moment_id: x for x in explanations}


def additivity_error(x: Explanation) -> float:
    """How far baseline + the three driver shap values + rest is from the raw score."""
    return abs(x.baseline + sum(d.shap for d in x.drivers) + x.rest - x.raw_score)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview Analyst Moments for one match.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args(argv)
    if not args.file.is_file():
        print(f"{args.file} not found, generating it with seed 42", file=sys.stderr)
        generator.main(["--seed", "42", "--out", str(args.file)])

    events = load_events(args.file)
    scorer = XPassScorer(*find_default_model())
    n_passes, moments = build_all_moments(events, scorer)

    explanations = explain_all(events, moments, scorer)

    counts = Counter(m.kind for m in moments)
    print(f"model {scorer.model_version}, {n_passes} passes, {len(moments)} moments")
    for kind in OUTLIER_KINDS:
        print(f"  {kind:<17} {counts.get(kind, 0)}")
    print(f"explained {len(explanations)} of {len(moments)} moments, "
          f"max additivity error {max((additivity_error(x) for x in explanations.values()), default=0.0):.3e}")
    n_drivers = Counter(len(x.drivers) for x in explanations.values())
    print("moments by driver count: " + ", ".join(f"{k} drivers {n_drivers.get(k, 0)}" for k in (3, 2, 1, 0)))
    for kind, top in select_highlights(moments, per_kind=1).items():
        print(f"\nstrongest {kind}:")
        print(top[0].model_dump_json(indent=2) if top else "(none)")
        if top:
            print(f"\nexplanation of the strongest {kind}:")
            print(explanations[top[0].moment_id].model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
