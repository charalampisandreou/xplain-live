"""
Offline pipeline: a match's events in, one MomentBundle (Moment + Explanation + commentary flag) per
outlier pass out. Deterministic code only: the Analyst and the Explainer, no LLM.

    python -m xplain.pipeline --file data/match_seed42.jsonl [--seed 42] [--full] [--timeline]
                              [--as-of-minute N] [--out data/moments_seed42.jsonl]

Every outlier gets a hover card; only some are commentated. The commentary gate (qualifies_live)
passes a moment when
  * "ranking": it enters the running top K (highlights_per_kind) of its kind in its half, counting
    only earlier moments of the same kind and period (the ranking resets at half-time), or
  * "threshold": it clears the commentary floor of its kind, whatever the ranking.
Later moments never change an earlier flag.

Reports: first_half and second_half hold the final top K of each kind in that period. Every report
moment was commentated live: a moment in the top K of a set is in the top K of any earlier subset
that contains it.

Timeline: every commentated moment so far, with its current place in the ranking of its kind and
half. Displaced and floor-only moments stay on the timeline with in_ranking False.
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from . import analytics, generator
from .analyst import OUTLIER_KINDS, build_moment, qualifies_live, select_half_highlights
from .config import AnalyticsConfig
from .contracts import (
    Explanation,
    Moment,
    MomentBundle,
    QualifiedBy,
    Report,
    TimelineItem,
)
from .explainer import ShapExplainer
from .schema import Event, MatchMetaEvent, PassEvent
from .scoring import XPassScorer, find_default_model

_EVENT = TypeAdapter(Event)
REPORT_SCOPES: tuple[tuple[str, int], ...] = (("first_half", 1), ("second_half", 2))


def ensure_match_file(path: Path, seed: int = 42) -> bool:
    """Generate the match file with the given seed when it does not exist yet; True if it was generated."""
    if path.is_file():
        return False
    print(f"{path} not found, generating it with seed {seed}", file=sys.stderr)
    generator.main(["--seed", str(seed), "--out", str(path)])
    return True


def load_events(path: Path) -> list:
    """Read a match JSONL file into schema events."""
    return [_EVENT.validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def names_from_meta(events: Sequence) -> tuple[dict[str, str], dict[str, str]]:
    """Player and team names from the match_meta event, as (player_id -> name, team_id -> name); empty without one."""
    meta = next((e for e in events if isinstance(e, MatchMetaEvent)), None)
    if meta is None:
        return {}, {}
    players = {p.player_id: p.name for p in meta.players}
    teams = {meta.home_team_id: meta.home_name, meta.away_team_id: meta.away_name}
    return players, teams


def additivity_error(x: Explanation) -> float:
    """How far baseline + the driver shap values + rest is from the raw score."""
    return abs(x.baseline + sum(d.shap for d in x.drivers) + x.rest - x.raw_score)


def clears_floor(moment: Moment, cfg: AnalyticsConfig) -> bool:
    """True when the moment is strong enough to be commentated whatever the ranking."""
    if moment.kind == "surprise_success":
        return moment.xp <= cfg.commentary_floor_success_max_xp
    return moment.xp >= cfg.commentary_floor_failure_min_xp


def commentary_reasons(moment: Moment, earlier: Sequence[Moment], cfg: AnalyticsConfig) -> tuple[QualifiedBy, ...]:
    """Why the moment is commentated: ("ranking",) / ("threshold",) / both / (). Uses only earlier moments of its period."""
    same_half = [m for m in earlier if m.period == moment.period]
    reasons: list[QualifiedBy] = []
    if qualifies_live(moment, same_half, cfg.highlights_per_kind):
        reasons.append("ranking")
    if clears_floor(moment, cfg):
        reasons.append("threshold")
    return tuple(reasons)


def process_match(events: Sequence, scorer: XPassScorer,
                  cfg: AnalyticsConfig = analytics.DEFAULT) -> list[MomentBundle]:
    """Score every pass, build Moments, run the commentary gate on earlier moments only, explain; in seq order."""
    passes = sorted((e for e in events if isinstance(e, PassEvent)), key=lambda e: e.seq)
    results = scorer.score([p.inputs for p in passes])
    players, teams = names_from_meta(events)

    pairs: list[tuple[PassEvent, Moment]] = []
    reasons: list[tuple[QualifiedBy, ...]] = []
    for event, result in zip(passes, results, strict=True):
        moment = build_moment(event, result.expected_probability, scorer.model_version, cfg=cfg,
                              player_names=players, team_names=teams)
        if moment is None:
            continue
        reasons.append(commentary_reasons(moment, [m for _, m in pairs], cfg))
        pairs.append((event, moment))

    explanations = ShapExplainer(scorer.boosters).explain_many(pairs)
    return [
        MomentBundle(moment=moment, explanation=explanation, qualifies_live=bool(why), qualified_by=why)
        for (_, moment), explanation, why in zip(pairs, explanations, reasons, strict=True)
    ]


def summary_line(model_version: str, n_passes: int, bundles: Sequence[MomentBundle]) -> str:
    """One line: model, passes, moments per kind, live per kind per half, threshold-only count, largest additivity error."""
    kinds = Counter(b.moment.kind for b in bundles)
    live = Counter((b.moment.kind, b.moment.period) for b in bundles if b.qualifies_live)
    floor_only = sum(b.qualified_by == ("threshold",) for b in bundles)
    max_err = max((additivity_error(b.explanation) for b in bundles), default=0.0)
    moments = ", ".join(f"{k} {kinds.get(k, 0)}" for k in OUTLIER_KINDS)
    live_part = ", ".join(f"{k} h1 {live.get((k, 1), 0)} h2 {live.get((k, 2), 0)}" for k in OUTLIER_KINDS)
    return (f"model {model_version}, {n_passes} passes, moments: {moments}, live: {live_part}, "
            f"threshold-only {floor_only}, max additivity error {max_err:.3e}")


def reports(bundles: Sequence[MomentBundle], per_kind: int) -> list[Report]:
    """The final ranking of each half: [first_half, second_half], top per_kind of each kind."""
    by_id = {b.moment.moment_id: b for b in bundles}
    top = select_half_highlights([b.moment for b in bundles], per_kind)
    return [
        Report(scope=scope, highlights={k: [by_id[m.moment_id] for m in ms] for k, ms in top[period].items()})
        for scope, period in REPORT_SCOPES
    ]


def timeline(bundles: Sequence[MomentBundle], per_kind: int, as_of_t: float | None = None) -> list[TimelineItem]:
    """Commentated moments with t <= as_of_t, in event order, ranked at as_of_t within their kind and half.

    rank is the place among ALL moments of the same kind and period with t <= as_of_t (the Analyst's
    sort rule); outside the top per_kind (displaced, or floor-only) in_ranking is False and rank None.
    """
    limit = math.inf if as_of_t is None else as_of_t
    so_far = sorted((b for b in bundles if b.moment.t <= limit), key=lambda b: b.moment.seq)
    ranked = select_half_highlights([b.moment for b in so_far], len(so_far))  # full ranking per half and kind
    rank_of = {m.moment_id: r for half in ranked.values() for ms in half.values() for r, m in enumerate(ms, start=1)}
    items = []
    for b in so_far:
        if not b.qualifies_live:
            continue
        rank = rank_of[b.moment.moment_id]
        in_ranking = rank <= per_kind
        items.append(TimelineItem(bundle=b, in_ranking=in_ranking, rank=rank if in_ranking else None))
    return items


def _why(b: MomentBundle) -> str:
    return "+".join(b.qualified_by) or "-"


def _player(m: Moment) -> str:
    return m.player_name or m.player_id or "-"


def report_table(reports_: Sequence[Report]) -> list[str]:
    """The reports as table rows: scope | kind | rank | minute | period | player | xP | delta | completed | qualified_by."""
    rows = ["scope       | kind             | rank | minute | period | player           | xP    | delta  | "
            "completed | qualified_by"]
    for report in reports_:
        for kind, bundles in report.highlights.items():
            if not bundles:
                rows.append(f"{report.scope:<11} | {kind:<16} | -    | (none)")
            for rank, b in enumerate(bundles, start=1):
                m = b.moment
                rows.append(f"{report.scope:<11} | {kind:<16} | {rank:<4} | {m.minute:<6} | {m.period:<6} | "
                            f"{_player(m):<16} | {m.xp:.3f} | {m.delta:+.3f} | {m.completed!s:<9} | {_why(b)}")
    return rows


def timeline_table(items: Sequence[TimelineItem]) -> list[str]:
    """The timeline as table rows: minute | kind | player | xP | delta | completed | qualified_by | in_ranking | rank."""
    rows = ["minute | kind             | player           | xP    | delta  | completed | qualified_by      | "
            "in_ranking | rank"]
    for item in items:
        b, m = item.bundle, item.bundle.moment
        rows.append(f"{m.minute:<6} | {m.kind:<16} | {_player(m):<16} | {m.xp:.3f} | {m.delta:+.3f} | "
                    f"{m.completed!s:<9} | {_why(b):<17} | {item.in_ranking!s:<10} | "
                    f"{item.rank if item.rank is not None else '-'}")
    return rows


def minute_to_t(minute: int) -> float:
    """as_of_t for "up to and including match minute N": the largest float below N * 60 seconds.

    Moment minute = floor(t / 60) + 1, so minute <= N exactly when t < N * 60. A moment at
    t = N * 60 is minute N + 1 and must not be included.
    """
    return math.nextafter(minute * 60.0, -math.inf)


def reports_path(out: Path) -> Path:
    """The reports file next to the bundles file: moments.jsonl -> moments_reports.jsonl."""
    return out.with_name(f"{out.stem}_reports{out.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Moments, explanations, per-half reports and timeline for a match.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used to generate --file when it does not exist")
    parser.add_argument("--full", action="store_true", help="also print every bundle as indented JSON")
    parser.add_argument("--timeline", action="store_true", help="print the timeline at the end of the match")
    parser.add_argument("--as-of-minute", type=int, metavar="N", help="print the timeline as it stood at match minute N")
    parser.add_argument("--out", type=Path, help="write one compact JSON line per bundle, and the reports to *_reports")
    args = parser.parse_args(argv)
    if args.as_of_minute is not None and args.as_of_minute < 0:
        parser.error("--as-of-minute must be >= 0")
    if ensure_match_file(args.file, args.seed):
        print(f"generated {args.file} with seed {args.seed}")

    cfg = analytics.DEFAULT
    events = load_events(args.file)
    scorer = XPassScorer(*find_default_model())
    bundles = process_match(events, scorer, cfg)
    match_reports = reports(bundles, cfg.highlights_per_kind)

    n_passes = sum(isinstance(e, PassEvent) for e in events)
    print(summary_line(scorer.model_version, n_passes, bundles))
    print("\n".join(report_table(match_reports)))
    if args.full:
        for b in bundles:
            print(b.model_dump_json(indent=2))
    if args.timeline:
        print("timeline at the end of the match")
        print("\n".join(timeline_table(timeline(bundles, cfg.highlights_per_kind))))
    if args.as_of_minute is not None:
        print(f"timeline at minute {args.as_of_minute}")
        print("\n".join(timeline_table(timeline(bundles, cfg.highlights_per_kind, minute_to_t(args.as_of_minute)))))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(b.model_dump_json() + "\n" for b in bundles), encoding="utf-8")
        reports_path(args.out).write_text("".join(r.model_dump_json() + "\n" for r in match_reports), encoding="utf-8")
        print(f"wrote {len(bundles)} bundles to {args.out} and {len(match_reports)} reports to "
              f"{reports_path(args.out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
