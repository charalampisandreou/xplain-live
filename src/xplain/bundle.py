"""
Match bundle: one JSON file per match that a static web page plays on the match clock.

    python -m xplain.bundle --file data/match_seed42.jsonl --out data/bundle_seed42.json
                            [--seed 42] [--mode strict|free|hints] [--no-llm] [--fresh]

Everything the page shows is computed here, by existing code; the page computes no numbers and no
minutes and calls no model:
  * clock:   one {t, minute_label} per match-minute boundary of each period, added time included,
             labelled by clock.match_clock; the page shows the label of the last entry with t <= now
  * marks:   every PassMark (pipeline.analyse_match) plus its display geometry: the pass drawn on
             one fixed pitch where the home team attacks right in the first half and left in the
             second (the away team the opposite); stored and model coordinates stay in the
             passer's own frame. meta.attacks_right names the team attacking right per half
  * moments: every outlier's MomentBundle, with its line (or None) and its narration record
  * reports: per half, the moment_ids of pipeline.reports
  * meta.players: every player of match_meta with team, position and shirt number
  * players: pipeline.player_summary at full time for all 22 players (None: no passes)
  * score:   0-0 at the first event, then one entry per goal (ShotEvent.goal), with the score after it
  * rankings: per commentated moment, in order, two snapshots of its kind as they stood at its t:
             scope "all" (the whole match so far) and scope "1" or "2" (its half). Each is the top K
             of all moments of that kind in the scope with t <= that t (analyst.select_highlights,
             the gate's sort), each entry with its rank in the previous snapshot of the same scope
             and kind (None: new); the page shows the latest snapshot with t <= now
  * tier:    each moment's difficulty tier words (narration.difficulty_tier), for the plate; each mark's
             tier words (narration.pass_tier, the same words on one xP scale) wherever a pass's difficulty is shown
  * recaps:  (schema "5", --recaps) per scope ("full_time", "half_time"): the recap card (recap.RecapCard, every
             number computed in code), the RecapEditor's verified text (or None) and how it came about
  * trace:   (schema "5") per commentated moment, its agent steps (trace.TraceStep: one per agent per
             attempt, Verifier reasons, model, latency of model calls); per recap scope, the RecapEditor's steps.
             For a later Analyst screen only; Studio does not show it.
             A schema "4" file (no trace, no recaps) still loads as a MatchBundle.

Narration: by default the Agent Framework workflow (workflow.run_match, needs az login). --no-llm
calls no model: commentated moments without a stored line are "not_run". A line is written once per
moment_id: when --out already holds a bundle of the same match_id and model_version (schema "1" or
"2"), its accepted lines are put into the Narrator's cache and the model is not called for them again; the final
Verifier check still runs on every reused line (a reused line keeps its stored Narrator steps in the
trace). --fresh ignores the old file.
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
from .analyst import select_highlights
from .analytics import Outlier
from .config import PHRASE_MODES, AnalyticsConfig, phrase_mode
from .contracts import MomentBundle, PassGeometry, PassMark, PlayerSummary
from .narration import NarratorOut, difficulty_tier, pass_tier
from .narrator import NarrationResult, Narrator
from .pipeline import analyse_match, ensure_match_file, load_events, player_summary, reports
from .recap import RECAP_SCOPES, RecapCard, RecapOut, slot_values, star_player, team_passing
from .recap_verifier import verify_recap
from .schema import PITCH_LENGTH, PITCH_WIDTH, MatchMetaEvent, ShotEvent
from .scoring import XPassScorer, find_default_model
from .trace import TraceStep, code_step, reused
from .verifier import verify
from .workflow import MomentOutcome, RecapOutcome, final_step, run_match, run_match_full

SCHEMA_VERSION = "5"
LOADABLE_SCHEMA_VERSIONS: tuple[str, ...] = ("4", "5")  # validate as a MatchBundle ("4": no trace)
OLD_SCHEMA_VERSIONS: tuple[str, ...] = ("1", "2", "3")  # still read for line reuse
RankingScope = Literal["all", "1", "2"]   # the whole match so far, or one half
MINUTE_S = 60.0
NarrationStatus = Literal["accepted", "retried", "rejected", "not_commentated", "not_run"]
LINE_STATUSES: tuple[str, ...] = ("accepted", "retried")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TeamRef(_Strict):
    team_id: str
    name: str


class PlayerRef(_Strict):
    """A player as in match_meta; position and shirt_number are None in bundles written before they were added."""

    player_id: str
    name: str
    team_id: str
    position: Literal["GK", "DEF", "MID", "FWD"] | None = None
    shirt_number: Annotated[int, Field(ge=1, le=99)] | None = None


class BundleMeta(_Strict):
    schema_version: Literal["4", "5"] = SCHEMA_VERSION
    match_id: str
    home: TeamRef
    away: TeamRef
    players: list[PlayerRef]
    model_version: str
    generator_version: str
    phrase_mode: str
    attacks_right: dict[Literal["1", "2"], str]   # half -> team_id attacking right on the display pitch


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
    """The top K of one kind in one scope as it stood at t (the t of the commentated moment that made it)."""

    t: float
    scope: RankingScope
    kind: Outlier
    entries: list[RankEntry]

    @model_validator(mode="after")
    def _check_ranks(self) -> RankingSnapshot:
        if [e.rank for e in self.entries] != list(range(1, len(self.entries) + 1)):
            raise ValueError(f"snapshot ranks must run 1..n in order, got {[e.rank for e in self.entries]}")
        return self


class DisplayGeometry(_Strict):
    """A pass on the display pitch (see display_geometry); for drawing only, never for the model."""

    start_x: float
    start_y: float
    end_x: float
    end_y: float


class BundleMark(PassMark):
    """A PassMark with its display geometry and its difficulty tier words (narration.pass_tier), which the page
    shows wherever a pass's difficulty is shown. tier is None in bundles written before it was added."""

    display: DisplayGeometry
    tier: Tier | None = None


def display_rotated(team_id: str, period: int, home_team_id: str) -> bool:
    """True when the team's passes are turned 180 degrees for display: the home team in the second
    half, the away team in the first (stored passes always attack toward x = 120, i.e. right)."""
    return (team_id == home_team_id) == (period == 2)


def display_geometry(g: PassGeometry, team_id: str, period: int, home_team_id: str) -> DisplayGeometry:
    """The pass on the display pitch: unchanged, or rotated 180 degrees, (x, y) -> (120 - x, 80 - y)."""
    if not display_rotated(team_id, period, home_team_id):
        return DisplayGeometry(start_x=g.start_x, start_y=g.start_y, end_x=g.end_x, end_y=g.end_y)
    return DisplayGeometry(start_x=PITCH_LENGTH - g.start_x, start_y=PITCH_WIDTH - g.start_y,
                           end_x=PITCH_LENGTH - g.end_x, end_y=PITCH_WIDTH - g.end_y)


def mark_tier(xp: float) -> Tier:
    """A pass's tier words from narration.pass_tier."""
    t = pass_tier(xp)
    return Tier(family=t.family, en=t.en, es=t.es, el=t.el)


def display_mark(mark: PassMark, home_team_id: str) -> BundleMark:
    return BundleMark(**dict(mark), display=display_geometry(mark.pass_geometry, mark.team_id, mark.period,
                                                             home_team_id), tier=mark_tier(mark.xp))


class RecapEntry(_Strict):
    """One recap: its card (numbers from code), its verified text or None, and how the text came about."""

    card: RecapCard
    text: RecapOut | None = None       # filled: the slots replaced by code from the card
    raw_text: RecapOut | None = None   # as the model wrote it, with slots
    slots: dict[str, str] = {}         # the slots used and their values
    status: Literal["accepted", "retried", "rejected"]
    attempts: Annotated[int, Field(ge=0)] = 0
    reasons: list[str] = []
    notes: list[str] = []
    latency_s: float = 0.0

    @model_validator(mode="after")
    def _check_status(self) -> RecapEntry:
        if (self.text is not None) != (self.status in LINE_STATUSES):
            raise ValueError(f"a recap text goes with status accepted or retried only (status {self.status})")
        return self


class Trace(_Strict):
    """The agent steps (trace.TraceStep) per commentated moment by moment_id, and per recap by scope."""

    moments: dict[str, list[TraceStep]] = {}
    recaps: dict[str, list[TraceStep]] = {}


class MatchBundle(_Strict):
    meta: BundleMeta
    clock: ClockTrack
    marks: list[BundleMark]
    moments: list[MomentEntry]
    reports: list[ReportIds]
    players: dict[str, PlayerSummary | None]
    score: Annotated[list[ScoreEntry], Field(min_length=1)]
    rankings: list[RankingSnapshot]
    recaps: dict[str, RecapEntry] = {}
    trace: Trace = Trace()

    @model_validator(mode="after")
    def _check_links(self) -> MatchBundle:
        ids = {m.bundle.moment.moment_id for m in self.moments}
        live_ids = {m.bundle.moment.moment_id for m in self.moments if m.bundle.qualifies_live}
        unknown = [i for i in self.trace.moments if i not in live_ids]
        if unknown:
            raise ValueError(f"trace names moments that are not commentated: {unknown}")
        if self.meta.schema_version == "4" and (self.trace.moments or self.trace.recaps or self.recaps):
            raise ValueError('a schema "4" bundle has no trace and no recaps')
        for scope, entry in self.recaps.items():
            if scope not in RECAP_SCOPES or entry.card.scope != scope or entry.card.match_id != self.meta.match_id:
                raise ValueError(f"recap {scope!r} does not fit its card ({entry.card.scope}, {entry.card.match_id})")
        unknown = [s for s in self.trace.recaps if s not in RECAP_SCOPES]
        if unknown:
            raise ValueError(f"trace names unknown recap scopes {unknown}")
        for r in self.reports:
            missing = [i for ids_ in r.highlights.values() for i in ids_ if i not in ids]
            if missing:
                raise ValueError(f"{r.scope} report names unknown moments {missing}")
        missing = [m.moment_id for m in self.marks if m.moment_id is not None and m.moment_id not in ids]
        if missing:
            raise ValueError(f"marks link unknown moments {missing}")
        by_id = {m.bundle.moment.moment_id: m.bundle.moment for m in self.moments}
        live = {m.bundle.moment.moment_id: m.bundle.qualifies_live for m in self.moments}
        for snap in self.rankings:
            for e in snap.entries:
                m = by_id.get(e.moment_id)
                if m is None:
                    raise ValueError(f"ranking snapshot at t={snap.t} names unknown moment {e.moment_id}")
                if (m.kind != snap.kind or m.t > snap.t or not in_scope(m.period, snap.scope)
                        or not live[e.moment_id]):
                    raise ValueError(f"ranking snapshot at t={snap.t} ({snap.kind}, scope {snap.scope}) holds "
                                     f"{e.moment_id} ({m.kind}, period {m.period}, t={m.t}, "
                                     f"commentated {live[e.moment_id]})")
        if [s.t for s in self.rankings] != sorted(s.t for s in self.rankings):
            raise ValueError("ranking snapshots must be in t order")
        home, away = self.meta.home.team_id, self.meta.away.team_id
        if self.meta.attacks_right != {"1": home, "2": away}:
            raise ValueError(f"attacks_right must be {{'1': {home!r}, '2': {away!r}}}, got {self.meta.attacks_right}")
        wrong = [m.event_id for m in self.marks
                 if m.display != display_geometry(m.pass_geometry, m.team_id, m.period, home)]
        if wrong:
            raise ValueError(f"marks with a wrong display geometry: {wrong[:5]}")
        wrong = [m.event_id for m in self.marks if m.tier is not None and m.tier != mark_tier(m.xp)]
        if wrong:
            raise ValueError(f"marks with a wrong tier: {wrong[:5]}")
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


def in_scope(period: int, scope: str) -> bool:
    return scope == "all" or scope == str(period)


def ranking_snapshots(bundles: Sequence[MomentBundle], per_kind: int) -> list[RankingSnapshot]:
    """Per commentated moment, in seq order, a snapshot of its kind for scope "all" and for its half:
    the top per_kind of all moments of that kind in the scope with t <= its t (select_highlights:
    |delta| first, ties to the earlier seq), each entry with its rank in the previous snapshot of
    the same scope and kind.

    Every entry is a commentated moment: a moment in the top per_kind of a set is in the top
    per_kind of every subset holding it, so also of its half when it happened (the gate's ranking).
    """
    moments = sorted((b.moment for b in bundles), key=lambda m: m.seq)
    previous: dict[tuple[str, str], dict[str, int]] = {}
    out: list[RankingSnapshot] = []
    for b in sorted((b for b in bundles if b.qualifies_live), key=lambda b: b.moment.seq):
        m = b.moment
        for scope in ("all", str(m.period)):
            eligible = [x for x in moments if x.t <= m.t and in_scope(x.period, scope)]
            top = select_highlights(eligible, per_kind)[m.kind]
            before = previous.get((scope, m.kind), {})
            entries = [RankEntry(moment_id=x.moment_id, rank=r, previous_rank=before.get(x.moment_id))
                       for r, x in enumerate(top, start=1)]
            previous[(scope, m.kind)] = {e.moment_id: e.rank for e in entries}
            out.append(RankingSnapshot(t=m.t, scope=scope, kind=m.kind, entries=entries))
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


def recap_entries(recaps: Sequence[RecapOutcome]) -> dict[str, RecapEntry]:
    """The recaps with a card, by scope (a recap whose card could not be built is in the trace only)."""
    return {r.scope: RecapEntry(card=r.card, text=r.text, raw_text=r.raw_text, slots=dict(r.slots),
                                status=r.status, attempts=r.attempts,
                                reasons=list(r.reasons), notes=list(r.notes), latency_s=r.latency_s)
            for r in recaps if r.card is not None}


def build_bundle(events: Sequence, scorer: XPassScorer, outcomes: Sequence[MomentOutcome] | None = None,
                 mode: str | None = None, cfg: AnalyticsConfig = analytics.DEFAULT,
                 recaps: Sequence[RecapOutcome] = ()) -> MatchBundle:
    """The whole match as one MatchBundle; workflow outcomes are merged into the moments by moment_id, and the
    recaps (workflow.RecapOutcome) go into recaps and trace.recaps by scope."""
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
            players=[PlayerRef(player_id=p.player_id, name=p.name, team_id=p.team_id, position=p.position,
                               shirt_number=p.shirt_number) for p in meta.players],
            model_version=scorer.model_version,
            generator_version=meta.generator_version,
            phrase_mode=mode or phrase_mode(),
            attacks_right={"1": meta.home_team_id, "2": meta.away_team_id},
        ),
        clock=clock_track(events),
        marks=[display_mark(m, meta.home_team_id) for m in marks],
        moments=moments,
        reports=report_ids,
        players={p.player_id: player_summary(marks, p.player_id) for p in meta.players},
        score=score_track(events, meta, names),
        rankings=ranking_snapshots(bundles, cfg.highlights_per_kind),
        recaps=recap_entries(recaps),
        trace=Trace(moments={b.moment.moment_id: list(o.trace) for b in bundles
                             if b.qualifies_live and (o := by_id.get(b.moment.moment_id)) is not None and o.trace},
                    recaps={r.scope: list(r.trace) for r in recaps if r.trace}),
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
                                                        latency_s=m.narration.latency_s,
                                                        trace=narrator_steps(old.trace, m.bundle.moment.moment_id,
                                                                             m.narration.attempts))
            for m in old.moments if m.line is not None}


def narrator_steps(trace: Trace | None, moment_id: str, attempts: int) -> list[TraceStep]:
    """The Narrator's own steps of a moment (its model calls and Verifier checks, not the final check), marked
    reused: what a reused line carries; the workflow adds the other steps again. With none stored (an older
    schema), one reused narrator step that says so."""
    steps = [s for s in (trace.moments.get(moment_id, []) if trace else [])
             if s.agent in ("narrator", "verifier") and not s.final]
    if not steps:
        steps = [TraceStep(agent="narrator", kind="llm", attempt=max(attempts, 1), ok=True,
                           notes=["line reused from an earlier bundle; its steps were not stored"])]
    return reused(steps)


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
                reasons=list(narration.reasons), notes=list(narration.verdict_notes), latency_s=narration.latency_s,
                trace=narrator_steps(None, m["bundle"]["moment"]["moment_id"], narration.attempts))
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
        trace = [code_step("analyst", notes=list(b.qualified_by)), code_step("explainer"), *result.trace,
                 final_step(verdict, result.attempts)]
        out.append(MomentOutcome(moment=b.moment, bundle=b, line=result.line if verdict.ok else None,
                                 verdict=verdict, attempts=result.attempts, reasons=reasons,
                                 latency_s=result.latency_s, trace=trace))
    return out


# --- CLI ---------------------------------------------------------------------------------------

def to_json(bundle: MatchBundle) -> str:
    return bundle.model_dump_json()


# the card fields added after bundles were first written; refresh_recap_cards fills them in
ADDED_CARD_FIELDS = {"expected_xp", "vs_expected_xp"}


def refresh_recap_cards(bundle: MatchBundle) -> MatchBundle:
    """The bundle with each recap card's passing and star recomputed from the bundle's own marks
    (recap.team_passing, recap.star_player), with no model call. Every field the card already had must come out
    the same; only ADDED_CARD_FIELDS are filled in. Each text is then filled again from its raw text (the model's,
    with slots) with the card's values (now two decimals) and checked again by recap_verifier.verify_recap; when
    the filled text changes, a recap_filler code step in the trace records it."""
    added = {"home": ADDED_CARD_FIELDS, "away": ADDED_CARD_FIELDS, "star": ADDED_CARD_FIELDS}
    players = [p.name for p in bundle.meta.players]
    recaps: dict[str, RecapEntry] = {}
    steps = dict(bundle.trace.recaps)
    for scope, entry in bundle.recaps.items():
        card = entry.card
        new = card.model_copy(update={
            "home": team_passing(bundle.marks, card.home.team_id, card.home.name, scope),
            "away": team_passing(bundle.marks, card.away.team_id, card.away.name, scope),
            "star": star_player(bundle.marks, scope)})
        if new.model_dump(exclude=added) != card.model_dump(exclude=added):
            raise ValueError(f"recap {scope!r}: the recomputed card differs from the stored one beyond the added fields")
        try:
            recaps[scope], step = refill_recap_text(entry.model_copy(update={"card": new}), players)
        except ValueError as exc:
            raise ValueError(f"recap {scope!r}: {exc}") from exc
        if step is not None:
            steps[scope] = [*steps.get(scope, []), step]
    return bundle.model_copy(update={"recaps": recaps, "trace": bundle.trace.model_copy(update={"recaps": steps})})


def refill_recap_text(entry: RecapEntry, players: Sequence[str]) -> tuple[RecapEntry, TraceStep | None]:
    """The entry with its text filled again from its raw text (with slots) and its card's values, checked again
    by recap_verifier.verify_recap (raises ValueError if it fails), and the recap_filler step that records a
    change (None when the filled text is the same, or the entry has no text)."""
    if entry.text is None or entry.raw_text is None:
        return entry, None
    verdict = verify_recap(entry.raw_text, entry.card, players)
    if not verdict.ok:
        raise ValueError(f"the refilled text fails recap_verifier: {verdict.reasons}")
    if verdict.filled == entry.text:
        return entry, None
    values = slot_values(entry.card)
    slots = {name: values[name] for name in verdict.slots_used}
    text = {lang: getattr(verdict.filled, f"text_{lang}") for lang in ("en", "es", "el")}
    notes = ["filled again by refresh_recap_cards with the card's two-decimal values"]
    step = code_step("recap_filler", attempt=entry.attempts, text=text,
                     notes=notes + [f"{{{k}}} = {v}" for k, v in slots.items()])
    return entry.model_copy(update={"text": verdict.filled, "slots": slots}), step


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


async def _narrate_online(events: Sequence, scorer: XPassScorer, mode: str, stored: dict[str, NarrationResult],
                          recap_scopes: Sequence[str] = ()) -> tuple[list[MomentOutcome], list[RecapOutcome]]:
    from azure.identity.aio import AzureCliCredential

    from .narrator import build_agent
    from .recap_editor import RecapEditor, build_recap_agent

    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential, mode=mode), mode=mode)
        if not recap_scopes:
            return await narrate_match(events, scorer, narrator, stored), []
        for moment_id, result in stored.items():
            narrator.remember(moment_id, result)
        run = await run_match_full(events, scorer, narrator, recap_editor=RecapEditor(build_recap_agent(credential)),
                                   recap_scopes=recap_scopes)
        return run.outcomes, run.recaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a match bundle: one JSON file the web page plays.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, help="seed used to generate --file when it does not exist")
    parser.add_argument("--mode", choices=PHRASE_MODES, help="phrase mode (default XPLAIN_PHRASE_MODE)")
    parser.add_argument("--no-llm", action="store_true", help="no model calls: commentated moments without a "
                                                               "stored line are not_run")
    parser.add_argument("--fresh", action="store_true", help="ignore the lines stored in --out")
    parser.add_argument("--recaps", nargs="+", choices=RECAP_SCOPES, default=[],
                        help="also write these recaps with the RecapEditor (needs the model; not with --no-llm)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ensure_match_file(args.file, args.seed)
    mode = args.mode or phrase_mode()
    events = load_events(args.file)
    scorer = _load_scorer()
    stored = {} if args.fresh else stored_lines(args.out, _meta(events).match_id, scorer.model_version)
    if stored:
        print(f"reusing {len(stored)} stored line(s) from {args.out}", file=sys.stderr)
    if args.no_llm and args.recaps:
        parser.error("--recaps needs the model; drop --no-llm")
    recaps: list[RecapOutcome] = []
    if args.no_llm:
        outcomes = offline_outcomes(events, scorer, stored, mode)
    else:
        outcomes, recaps = asyncio.run(_narrate_online(events, scorer, mode, stored, args.recaps))
    bundle = build_bundle(events, scorer, outcomes, mode, recaps=recaps)
    text = to_json(bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(summary_line(bundle, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
