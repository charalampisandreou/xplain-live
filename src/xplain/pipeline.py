"""
Offline pipeline: a match's events in, one MomentBundle (Moment + Explanation + commentary flag) per
outlier pass out. Deterministic code only: the Analyst and the Explainer, no LLM.

    python -m xplain.pipeline --file data/match_seed42.jsonl [--seed 42] [--full] [--timeline]
                              [--as-of-minute N] [--out data/moments_seed42.jsonl]
                              [--marks-out data/marks_seed42.jsonl] [--player PLAYER_ID]

Every outlier gets a hover card; only some are commentated. The commentary gate (qualifies_live)
passes a moment when
  * "ranking": it enters the running top K (highlights_per_kind) of its kind in its half, counting
    only earlier moments of the same kind and period (the ranking resets at half-time), or
  * "threshold": it clears the commentary floor of its kind, whatever the ranking.
Later moments never change an earlier flag.

Reports: first_half and second_half hold the final top K of each kind in that period. Every report
moment was commentated live: a moment in the top K of a set is in the top K of any earlier subset
that contains it.

Pass marks: a light record of EVERY pass (geometry, xP, delta = PVA, difficulty, band, outlier
link), for pass vectors, hover details and per-player pass maps.

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

from . import analytics, clock, generator
from .analyst import OUTLIER_KINDS, build_moment, qualifies_live, select_half_highlights
from .config import AnalyticsConfig
from .contracts import (
    Explanation,
    Moment,
    MomentBundle,
    PassGeometry,
    PassMark,
    PlayerSummary,
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


class _ScoredMatch:
    """Every pass scored once, in seq order, with its Moment when it is an outlier. Shared by bundles and marks."""

    def __init__(self, events: Sequence, scorer: XPassScorer, cfg: AnalyticsConfig):
        passes = sorted((e for e in events if isinstance(e, PassEvent)), key=lambda e: e.seq)
        results = scorer.score([p.inputs for p in passes])
        self.players, self.teams = names_from_meta(events)
        self.second_half_start = clock.second_half_start(events)
        self.scored: list[tuple[PassEvent, float, Moment | None]] = []
        for event, result in zip(passes, results, strict=True):
            moment = build_moment(event, result.expected_probability, scorer.model_version, cfg=cfg,
                                  player_names=self.players, team_names=self.teams,
                                  second_half_start=self.second_half_start)
            self.scored.append((event, result.expected_probability, moment))
        self.cfg, self.scorer = cfg, scorer

    def bundles(self) -> list[MomentBundle]:
        """Run the commentary gate on earlier moments only and explain every Moment; in seq order."""
        pairs: list[tuple[PassEvent, Moment]] = []
        reasons: list[tuple[QualifiedBy, ...]] = []
        for event, _, moment in self.scored:
            if moment is None:
                continue
            reasons.append(commentary_reasons(moment, [m for _, m in pairs], self.cfg))
            pairs.append((event, moment))
        explanations = ShapExplainer(self.scorer.boosters).explain_many(pairs)
        return [
            MomentBundle(moment=moment, explanation=explanation, qualifies_live=bool(why), qualified_by=why)
            for (_, moment), explanation, why in zip(pairs, explanations, reasons, strict=True)
        ]

    def marks(self) -> list[PassMark]:
        """One PassMark per pass, in seq order, linked to its Moment when it is an outlier."""
        return [build_pass_mark(event, xp, moment, self.cfg, self.players, self.teams, self.second_half_start)
                for event, xp, moment in self.scored]


def build_pass_mark(event: PassEvent, xp: float, moment: Moment | None, cfg: AnalyticsConfig,
                    player_names: dict[str, str], team_names: dict[str, str],
                    second_half_start: float | None = None) -> PassMark:
    """The light record of one scored pass; difficulty, band and delta come from analytics.py, the minute from clock.py."""
    completed = event.details.completed
    d = analytics.difficulty(xp)
    c = clock.match_clock(event.t, event.period, second_half_start)
    return PassMark(
        event_id=event.event_id,
        seq=event.seq,
        t=event.t,
        minute=c.minute,
        minute_added=c.added,
        minute_label=c.label,
        period=event.period,
        team_id=event.team_id,
        team_name=team_names.get(event.team_id),
        player_id=event.player_id,
        player_name=player_names.get(event.player_id) if event.player_id is not None else None,
        pass_geometry=PassGeometry(
            start_x=event.inputs.start_x, start_y=event.inputs.start_y,
            end_x=event.details.end_x, end_y=event.details.end_y,
            length=event.details.length, ball_speed_ms=event.details.ball_speed_ms,
        ),
        completed=completed,
        xp=xp,
        delta=analytics.delta(completed, xp),
        difficulty=d,
        band=analytics.difficulty_band(d, cfg),
        outlier=moment.kind if moment is not None else None,
        moment_id=moment.moment_id if moment is not None else None,
    )


def analyse_match(events: Sequence, scorer: XPassScorer,
                  cfg: AnalyticsConfig = analytics.DEFAULT) -> tuple[list[MomentBundle], list[PassMark]]:
    """Bundles (outliers) and marks (every pass) from a single scoring of the passes."""
    match = _ScoredMatch(events, scorer, cfg)
    return match.bundles(), match.marks()


def process_match(events: Sequence, scorer: XPassScorer,
                  cfg: AnalyticsConfig = analytics.DEFAULT) -> list[MomentBundle]:
    """Score every pass, build Moments, run the commentary gate on earlier moments only, explain; in seq order."""
    return _ScoredMatch(events, scorer, cfg).bundles()


def pass_marks(events: Sequence, scorer: XPassScorer, cfg: AnalyticsConfig = analytics.DEFAULT) -> list[PassMark]:
    """One PassMark per PassEvent, in seq order, with the outlier link for passes that become Moments. No SHAP."""
    return _ScoredMatch(events, scorer, cfg).marks()


def player_pass_map(marks: Sequence[PassMark], player_id: str, as_of_t: float | None = None) -> list[PassMark]:
    """The player's passes with t <= as_of_t (all when None), in seq order. Passes without a player never match."""
    if player_id is None:
        return []
    limit = math.inf if as_of_t is None else as_of_t
    return sorted((m for m in marks if m.player_id == player_id and m.t <= limit), key=lambda m: m.seq)


def player_summary(marks: Sequence[PassMark], player_id: str, as_of_t: float | None = None) -> PlayerSummary | None:
    """Counts, completion rate, mean xP and PVA of the player's passes so far; None when there are none."""
    own = player_pass_map(marks, player_id, as_of_t)
    if not own:
        return None
    n = len(own)
    done = sum(m.completed for m in own)
    pva = sum(m.delta for m in own)
    return PlayerSummary(
        player_id=player_id,
        player_name=own[0].player_name,
        team_id=own[0].team_id,
        passes=n,
        completed=done,
        completion_rate=done / n,
        mean_xp=sum(m.xp for m in own) / n,
        pva_total=pva,
        pva_per_pass=pva / n,
        outliers=sum(m.outlier is not None for m in own),
    )


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
                rows.append(f"{report.scope:<11} | {kind:<16} | {rank:<4} | {m.minute_label:<6} | {m.period:<6} | "
                            f"{_player(m):<16} | {m.xp:.3f} | {m.delta:+.3f} | {m.completed!s:<9} | {_why(b)}")
    return rows


def timeline_table(items: Sequence[TimelineItem]) -> list[str]:
    """The timeline as table rows: minute | kind | player | xP | delta | completed | qualified_by | in_ranking | rank."""
    rows = ["minute | kind             | player           | xP    | delta  | completed | qualified_by      | "
            "in_ranking | rank"]
    for item in items:
        b, m = item.bundle, item.bundle.moment
        rows.append(f"{m.minute_label:<6} | {m.kind:<16} | {_player(m):<16} | {m.xp:.3f} | {m.delta:+.3f} | "
                    f"{m.completed!s:<9} | {_why(b):<17} | {item.in_ranking!s:<10} | "
                    f"{item.rank if item.rank is not None else '-'}")
    return rows


def reports_path(out: Path) -> Path:
    """The reports file next to the bundles file: moments.jsonl -> moments_reports.jsonl."""
    return out.with_name(f"{out.stem}_reports{out.suffix}")


def marks_path(out: Path) -> Path:
    """The marks file next to the bundles file: moments.jsonl -> moments_marks.jsonl."""
    return out.with_name(f"{out.stem}_marks{out.suffix}")


def player_line(s: PlayerSummary) -> str:
    """One line for a player's summary."""
    return (f"player {s.player_id} ({s.player_name or '-'}, {s.team_id}): {s.passes} passes, {s.completed} completed "
            f"({s.completion_rate:.3f}), mean xP {s.mean_xp:.3f}, PVA {s.pva_total:+.3f} ({s.pva_per_pass:+.4f} per pass), "
            f"{s.outliers} outliers")


def player_table(marks: Sequence[PassMark]) -> list[str]:
    """A player's passes as table rows: minute | start -> end | xP | delta | completed | outlier."""
    rows = ["minute | start -> end                 | xP    | delta  | completed | outlier"]
    for m in marks:
        g = m.pass_geometry
        vector = f"({g.start_x:5.1f}, {g.start_y:4.1f}) -> ({g.end_x:5.1f}, {g.end_y:4.1f})"
        rows.append(f"{m.minute_label:<6} | {vector:<28} | {m.xp:.3f} | {m.delta:+.3f} | {m.completed!s:<9} | {m.outlier or '-'}")
    return rows


def write_jsonl(path: Path, items: Sequence) -> None:
    """One compact JSON line per pydantic model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Moments, explanations, per-half reports and timeline for a match.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used to generate --file when it does not exist")
    parser.add_argument("--full", action="store_true", help="also print every bundle as indented JSON")
    parser.add_argument("--timeline", action="store_true", help="print the timeline at the end of the match")
    parser.add_argument("--as-of-minute", type=int, metavar="N", help="print the timeline as it stood at match minute N")
    parser.add_argument("--out", type=Path,
                        help="write one compact JSON line per bundle, the reports to *_reports and the pass marks to *_marks")
    parser.add_argument("--marks-out", type=Path, help="write one compact JSON line per pass (PassMark)")
    parser.add_argument("--player", metavar="PLAYER_ID", help="print the player's summary and passes (up to --as-of-minute)")
    args = parser.parse_args(argv)
    if args.as_of_minute is not None and args.as_of_minute < 0:
        parser.error("--as-of-minute must be >= 0")
    if ensure_match_file(args.file, args.seed):
        print(f"generated {args.file} with seed {args.seed}")

    cfg = analytics.DEFAULT
    events = load_events(args.file)
    scorer = XPassScorer(*find_default_model())
    bundles, marks = analyse_match(events, scorer, cfg)
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
        print("\n".join(timeline_table(timeline(bundles, cfg.highlights_per_kind, clock.as_of_t(events, args.as_of_minute)))))
    if args.player is not None:
        as_of = clock.as_of_t(events, args.as_of_minute) if args.as_of_minute is not None else None
        summary = player_summary(marks, args.player, as_of)
        print(player_line(summary) if summary else f"player {args.player}: no passes")
        print("\n".join(player_table(player_pass_map(marks, args.player, as_of))))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(b.model_dump_json() + "\n" for b in bundles), encoding="utf-8")
        reports_path(args.out).write_text("".join(r.model_dump_json() + "\n" for r in match_reports), encoding="utf-8")
        write_jsonl(marks_path(args.out), marks)
        print(f"wrote {len(bundles)} bundles to {args.out} and {len(match_reports)} reports to "
              f"{reports_path(args.out)}", file=sys.stderr)
        print(f"wrote {len(marks)} pass marks to {marks_path(args.out)}", file=sys.stderr)
    if args.marks_out:
        write_jsonl(args.marks_out, marks)
        print(f"wrote {len(marks)} pass marks to {args.marks_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
