"""
Match bundle: one JSON file per match that a static web page plays on the match clock.

    python -m xplain.bundle --file data/match_seed42.jsonl --out data/bundle_seed42.json
                            [--seed 42] [--mode strict|free|hints] [--no-llm] [--fresh]

Everything the page shows is computed here, by existing code; the page computes no numbers and no
minutes and calls no model:
  * clock:   one {t, minute_label} per match-minute boundary of each period, added time included,
             labelled by clock.match_clock; the page shows the label of the last entry with t <= now
  * marks:   every PassMark (pipeline.analyse_match)
  * moments: every outlier's MomentBundle, with its line (or None) and its narration record
  * reports: per half, the moment_ids of pipeline.reports
  * players: pipeline.player_summary at full time for all 22 players (None: no passes)
  * score:   0-0 at the first event, then one entry per goal (ShotEvent.goal), with the score after it
  * rankings: one snapshot per commentated moment, in order: the top K of its kind and half as it
             stood at its t (pipeline.timeline), each entry with its rank in the previous snapshot of
             the same kind and half (None: new); the page shows the latest snapshot with t <= now
  * tier:    each moment's difficulty tier words (narration.difficulty_tier), for the plate

Narration: by default the Agent Framework workflow (workflow.run_match, needs az login). --no-llm
calls no model: commentated moments without a stored line are "not_run". A line is written once per
moment_id: when --out already holds a bundle of the same match_id and model_version (schema "1" or
"2"), its accepted lines are put into the Narrator's cache and the model is not called for them again; the final
Verifier check still runs on every reused line. --fresh ignores the old file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import analytics, clock
from .analytics import Outlier
from .config import PHRASE_MODES, AnalyticsConfig, phrase_mode
from .contracts import MomentBundle, PassMark, PlayerSummary
from .narration import NarratorOut, difficulty_tier
from .narrator import NarrationResult, Narrator
from .pipeline import analyse_match, ensure_match_file, load_events, player_summary, reports, timeline
from .schema import MatchMetaEvent, ShotEvent
from .scoring import XPassScorer, find_default_model
from .verifier import verify
from .workflow import MomentOutcome, run_match

SCHEMA_VERSION = "2"
OLD_SCHEMA_VERSIONS: tuple[str, ...] = ("1",)  # still read for line reuse
MINUTE_S = 60.0
NarrationStatus = Literal["accepted", "retried", "rejected", "not_commentated", "not_run"]
LINE_STATUSES: tuple[str, ...] = ("accepted", "retried")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TeamRef(_Strict):
    team_id: str
    name: str


class PlayerRef(_Strict):
    player_id: str
    name: str
    team_id: str


class BundleMeta(_Strict):
    schema_version: Literal["2"] = SCHEMA_VERSION
    match_id: str
    home: TeamRef
    away: TeamRef
    players: list[PlayerRef]
    model_version: str
    generator_version: str
    phrase_mode: str


class ClockTick(_Strict):
    t: float
    minute_label: str


class ClockTrack(_Strict):
    """Minute boundaries in t order; the page shows the minute_label of the last tick with t <= now."""

    first_event_t: float
    second_half_start: float | None   # None when the match has no second half
    last_event_t: float
    ticks: list[ClockTick]


class Narration(_Strict):
    """How the moment's line came about. attempts 0: the model was not called for this moment."""

    status: NarrationStatus
    attempts: Annotated[int, Field(ge=0)] = 0
    reasons: list[str] = []
    latency_s: float = 0.0
    verdict_notes: list[str] = []


class Tier(_Strict):
    """The moment's difficulty tier words, copied from narration.DIFFICULTY_TIERS."""

    family: Literal["easy", "hard"]
    en: str
    es: str
    el: str


class MomentEntry(_Strict):
    bundle: MomentBundle
    tier: Tier
    line: NarratorOut | None = None
    narration: Narration

    @model_validator(mode="after")
    def _check_status(self) -> MomentEntry:
        status = self.narration.status
        if (self.line is not None) != (status in LINE_STATUSES):
            raise ValueError(f"a line goes with status accepted or retried only (status {status}, "
                             f"line {'set' if self.line else 'None'})")
        if (status == "not_commentated") == self.bundle.qualifies_live:
            raise ValueError(f"status not_commentated must match qualifies_live False (status {status})")
        return self


class ReportIds(_Strict):
    """A half's report as moment_ids per kind, in report order; the moments themselves are in MatchBundle.moments."""

    scope: Literal["first_half", "second_half"]
    highlights: dict[Outlier, list[str]]

    @model_validator(mode="after")
    def _check_kinds(self) -> ReportIds:
        if set(self.highlights) != set(get_args(Outlier)):
            raise ValueError(f"highlights must have exactly the keys {get_args(Outlier)}")
        return self


class Goal(_Strict):
    team_id: str
    player_id: str | None = None
    player_name: str | None = None


class ScoreEntry(_Strict):
    """The score from t on; goal is None for the opening 0-0 entry."""

    t: float
    minute_label: str
    home: Annotated[int, Field(ge=0)]
    away: Annotated[int, Field(ge=0)]
    goal: Goal | None = None


class RankEntry(_Strict):
    moment_id: str
    rank: Annotated[int, Field(ge=1)]
    previous_rank: Annotated[int, Field(ge=1)] | None = None   # None: not in the previous snapshot (new)


class RankingSnapshot(_Strict):
    """The top K of one kind and half as it stood at t (the t of the commentated moment that made it)."""

    t: float
    period: Literal[1, 2]
    kind: Outlier
    entries: list[RankEntry]

    @model_validator(mode="after")
    def _check_ranks(self) -> RankingSnapshot:
        if [e.rank for e in self.entries] != list(range(1, len(self.entries) + 1)):
            raise ValueError(f"snapshot ranks must run 1..n in order, got {[e.rank for e in self.entries]}")
        return self


class MatchBundle(_Strict):
    meta: BundleMeta
    clock: ClockTrack
    marks: list[PassMark]
    moments: list[MomentEntry]
    reports: list[ReportIds]
    players: dict[str, PlayerSummary | None]
    score: Annotated[list[ScoreEntry], Field(min_length=1)]
    rankings: list[RankingSnapshot]

    @model_validator(mode="after")
    def _check_links(self) -> MatchBundle:
        ids = {m.bundle.moment.moment_id for m in self.moments}
        for r in self.reports:
            missing = [i for ids_ in r.highlights.values() for i in ids_ if i not in ids]
            if missing:
                raise ValueError(f"{r.scope} report names unknown moments {missing}")
        missing = [m.moment_id for m in self.marks if m.moment_id is not None and m.moment_id not in ids]
        if missing:
            raise ValueError(f"marks link unknown moments {missing}")
        by_id = {m.bundle.moment.moment_id: m.bundle.moment for m in self.moments}
        for snap in self.rankings:
            for e in snap.entries:
                m = by_id.get(e.moment_id)
                if m is None:
                    raise ValueError(f"ranking snapshot at t={snap.t} names unknown moment {e.moment_id}")
                if (m.kind, m.period) != (snap.kind, snap.period) or m.t > snap.t:
                    raise ValueError(f"ranking snapshot at t={snap.t} ({snap.kind}, period {snap.period}) "
                                     f"holds {e.moment_id} ({m.kind}, period {m.period}, t={m.t})")
        if [s.t for s in self.rankings] != sorted(s.t for s in self.rankings):
            raise ValueError("ranking snapshots must be in t order")
        return self


# --- building ----------------------------------------------------------------------------------

def _meta(events: Sequence) -> MatchMetaEvent:
    meta = next((e for e in events if isinstance(e, MatchMetaEvent)), None)
    if meta is None:
        raise ValueError("a match bundle needs the match_meta event")
    return meta


def clock_track(events: Sequence) -> ClockTrack:
    """A tick at every minute boundary of each period, from its kick-off to its last event, labelled by match_clock."""
    start = clock.second_half_start(events)
    ticks: list[ClockTick] = []
    for period, kick_off in ((1, 0.0), (2, start)):
        times = [e.t for e in events if e.period == period]
        if not times or kick_off is None:
            continue
        k = 0
        while (t := round(kick_off + k * MINUTE_S, 3)) <= max(times):
            ticks.append(ClockTick(t=t, minute_label=clock.match_clock(t, period, start).label))
            k += 1
    all_t = [e.t for e in events]
    return ClockTrack(first_event_t=min(all_t), second_half_start=start, last_event_t=max(all_t), ticks=ticks)


def score_track(events: Sequence, meta: MatchMetaEvent, player_names: dict[str, str]) -> list[ScoreEntry]:
    """0-0 at the first event, then the score after each goal (ShotEvent.goal), in event order."""
    start = clock.second_half_start(events)
    first = min(events, key=lambda e: (e.t, e.seq))
    out = [ScoreEntry(t=first.t, minute_label=clock.match_clock(first.t, first.period, start).label, home=0, away=0)]
    home = away = 0
    for shot in sorted((e for e in events if isinstance(e, ShotEvent) and e.goal), key=lambda e: e.seq):
        if shot.team_id == meta.home_team_id:
            home += 1
        else:
            away += 1
        goal = Goal(team_id=shot.team_id, player_id=shot.player_id,
                    player_name=player_names.get(shot.player_id) if shot.player_id else None)
        out.append(ScoreEntry(t=shot.t, minute_label=clock.match_clock(shot.t, shot.period, start).label,
                              home=home, away=away, goal=goal))
    return out


def ranking_snapshots(bundles: Sequence[MomentBundle], per_kind: int) -> list[RankingSnapshot]:
    """One snapshot per commentated moment, in seq order: the top per_kind of its kind and half at its t
    (pipeline.timeline), with each entry's rank in the previous snapshot of the same kind and half."""
    previous: dict[tuple[str, int], dict[str, int]] = {}
    out: list[RankingSnapshot] = []
    for b in sorted((b for b in bundles if b.qualifies_live), key=lambda b: b.moment.seq):
        m = b.moment
        items = sorted((i for i in timeline(bundles, per_kind, m.t)
                        if i.in_ranking and (i.bundle.moment.kind, i.bundle.moment.period) == (m.kind, m.period)),
                       key=lambda i: i.rank)
        before = previous.get((m.kind, m.period), {})
        entries = [RankEntry(moment_id=i.bundle.moment.moment_id, rank=i.rank,
                             previous_rank=before.get(i.bundle.moment.moment_id)) for i in items]
        previous[(m.kind, m.period)] = {e.moment_id: e.rank for e in entries}
        out.append(RankingSnapshot(t=m.t, period=m.period, kind=m.kind, entries=entries))
    return out


def tier_of(bundle: MomentBundle) -> Tier:
    """The moment's tier words from narration.difficulty_tier."""
    t = difficulty_tier(bundle.moment)
    return Tier(family=t.family, en=t.en, es=t.es, el=t.el)


def _narration(bundle: MomentBundle, outcome: MomentOutcome | None) -> tuple[NarratorOut | None, Narration]:
    if not bundle.qualifies_live:
        return None, Narration(status="not_commentated")
    if outcome is None:
        return None, Narration(status="not_run")
    return outcome.line, Narration(status=outcome.status, attempts=outcome.attempts, reasons=list(outcome.reasons),
                                   latency_s=outcome.latency_s,
                                   verdict_notes=list(outcome.verdict.notes) if outcome.verdict else [])


def build_bundle(events: Sequence, scorer: XPassScorer, outcomes: Sequence[MomentOutcome] | None = None,
                 mode: str | None = None, cfg: AnalyticsConfig = analytics.DEFAULT) -> MatchBundle:
    """The whole match as one MatchBundle; workflow outcomes are merged into the moments by moment_id."""
    meta = _meta(events)
    bundles, marks = analyse_match(events, scorer, cfg)
    by_id = {o.moment.moment_id: o for o in outcomes or ()}
    moments = []
    for b in bundles:
        line, narration = _narration(b, by_id.get(b.moment.moment_id))
        moments.append(MomentEntry(bundle=b, tier=tier_of(b), line=line, narration=narration))
    report_ids = [ReportIds(scope=r.scope, highlights={k: [b.moment.moment_id for b in bs]
                                                        for k, bs in r.highlights.items()})
                  for r in reports(bundles, cfg.highlights_per_kind)]
    names = {p.player_id: p.name for p in meta.players}
    return MatchBundle(
        meta=BundleMeta(
            match_id=meta.match_id,
            home=TeamRef(team_id=meta.home_team_id, name=meta.home_name),
            away=TeamRef(team_id=meta.away_team_id, name=meta.away_name),
            players=[PlayerRef(player_id=p.player_id, name=p.name, team_id=p.team_id) for p in meta.players],
            model_version=scorer.model_version,
            generator_version=meta.generator_version,
            phrase_mode=mode or phrase_mode(),
        ),
        clock=clock_track(events),
        marks=marks,
        moments=moments,
        reports=report_ids,
        players={p.player_id: player_summary(marks, p.player_id) for p in meta.players},
        score=score_track(events, meta, names),
        rankings=ranking_snapshots(bundles, cfg.highlights_per_kind),
    )


# --- line reuse --------------------------------------------------------------------------------

def stored_lines(path: Path, match_id: str, model_version: str) -> dict[str, NarrationResult]:
    """The accepted lines of an earlier bundle of the same match and model, by moment_id; {} otherwise.

    A current bundle is validated whole; an older schema (OLD_SCHEMA_VERSIONS) is read for its lines only.
    """
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        old = MatchBundle.model_validate_json(text)
    except (ValidationError, ValueError):
        return _old_schema_lines(text, match_id, model_version)
    if old.meta.match_id != match_id or old.meta.model_version != model_version:
        return {}
    return {m.bundle.moment.moment_id: NarrationResult(line=m.line, attempts=m.narration.attempts,
                                                        reasons=list(m.narration.reasons),
                                                        notes=list(m.narration.verdict_notes),
                                                        latency_s=m.narration.latency_s)
            for m in old.moments if m.line is not None}


def _old_schema_lines(text: str, match_id: str, model_version: str) -> dict[str, NarrationResult]:
    """The lines of a bundle in an older schema: each line and its narration record validated, the rest ignored."""
    try:
        raw = json.loads(text)
        meta = raw["meta"]
        if (meta.get("schema_version") not in OLD_SCHEMA_VERSIONS or meta.get("match_id") != match_id
                or meta.get("model_version") != model_version):
            return {}
        out = {}
        for m in raw["moments"]:
            if m.get("line") is None:
                continue
            narration = Narration.model_validate(m["narration"])
            if narration.status not in LINE_STATUSES:
                continue
            out[m["bundle"]["moment"]["moment_id"]] = NarrationResult(
                line=NarratorOut.model_validate(m["line"]), attempts=narration.attempts,
                reasons=list(narration.reasons), notes=list(narration.verdict_notes), latency_s=narration.latency_s)
        return out
    except (ValueError, KeyError, TypeError, AttributeError):
        return {}


async def narrate_match(events: Sequence, scorer: XPassScorer, narrator: Narrator,
                        stored: dict[str, NarrationResult] | None = None,
                        cfg: AnalyticsConfig = analytics.DEFAULT) -> list[MomentOutcome]:
    """workflow.run_match, with the stored results in the Narrator's cache (no model call for them)."""
    for moment_id, result in (stored or {}).items():
        narrator.remember(moment_id, result)
    return await run_match(events, scorer, narrator, cfg=cfg)


def offline_outcomes(events: Sequence, scorer: XPassScorer, stored: dict[str, NarrationResult], mode: str,
                     cfg: AnalyticsConfig = analytics.DEFAULT) -> list[MomentOutcome]:
    """--no-llm: an outcome only for commentated moments with a stored line, after the same final
    Verifier check as the workflow's VerifierExecutor. The other commentated moments stay not_run."""
    bundles, _ = analyse_match(events, scorer, cfg)
    out = []
    for b in bundles:
        result = stored.get(b.moment.moment_id)
        if not b.qualifies_live or result is None:
            continue
        verdict = verify(result.line, b, mode)
        reasons = list(result.reasons) + [f"final check: {r}" for r in verdict.reasons]
        out.append(MomentOutcome(moment=b.moment, bundle=b, line=result.line if verdict.ok else None,
                                 verdict=verdict, attempts=result.attempts, reasons=reasons,
                                 latency_s=result.latency_s))
    return out


# --- CLI ---------------------------------------------------------------------------------------

def to_json(bundle: MatchBundle) -> str:
    return bundle.model_dump_json()


def summary_line(bundle: MatchBundle, size_bytes: int) -> str:
    """passes, moments, commentated, narration statuses, goals and final score, file size."""
    statuses = [m.narration.status for m in bundle.moments]
    count = {s: statuses.count(s) for s in ("accepted", "retried", "rejected", "not_run")}
    final = bundle.score[-1]
    return (f"{len(bundle.marks)} passes, {len(bundle.moments)} moments, "
            f"{sum(m.bundle.qualifies_live for m in bundle.moments)} commentated: accepted {count['accepted']}, "
            f"retried {count['retried']}, rejected {count['rejected']}, not_run {count['not_run']}; "
            f"goals {len(bundle.score) - 1}, final score {bundle.meta.home.name} {final.home}-{final.away} "
            f"{bundle.meta.away.name}; {size_bytes} bytes ({size_bytes / 1024:.0f} KB)")


def _load_scorer() -> XPassScorer:
    return XPassScorer(*find_default_model())


async def _narrate_online(events: Sequence, scorer: XPassScorer, mode: str,
                          stored: dict[str, NarrationResult]) -> list[MomentOutcome]:
    from azure.identity.aio import AzureCliCredential

    from .narrator import build_agent

    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential, mode=mode), mode=mode)
        return await narrate_match(events, scorer, narrator, stored)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a match bundle: one JSON file the web page plays.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used to generate --file when it does not exist")
    parser.add_argument("--mode", choices=PHRASE_MODES, help="phrase mode (default XPLAIN_PHRASE_MODE)")
    parser.add_argument("--no-llm", action="store_true", help="no model calls: commentated moments without a "
                                                               "stored line are not_run")
    parser.add_argument("--fresh", action="store_true", help="ignore the lines stored in --out")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ensure_match_file(args.file, args.seed)
    mode = args.mode or phrase_mode()
    events = load_events(args.file)
    scorer = _load_scorer()
    stored = {} if args.fresh else stored_lines(args.out, _meta(events).match_id, scorer.model_version)
    if stored:
        print(f"reusing {len(stored)} stored line(s) from {args.out}", file=sys.stderr)
    if args.no_llm:
        outcomes = offline_outcomes(events, scorer, stored, mode)
    else:
        outcomes = asyncio.run(_narrate_online(events, scorer, mode, stored))
    bundle = build_bundle(events, scorer, outcomes, mode)
    text = to_json(bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(summary_line(bundle, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
