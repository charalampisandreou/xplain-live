import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from xplain import analytics, pipeline
from xplain.analyst import qualifies_live, select_half_highlights, select_highlights
from xplain.config import AnalyticsConfig
from xplain.contracts import MomentBundle, Report, TimelineItem
from xplain.features import FEATURE_ORDER
from xplain.generator import generate_match
from xplain.pipeline import (
    additivity_error,
    clears_floor,
    ensure_match_file,
    load_events,
    minute_to_t,
    names_from_meta,
    process_match,
    reports,
    reports_path,
    summary_line,
    timeline,
)
from xplain.schema import BodyPart, MatchMetaEvent, PassDetails, PassEvent, PassHeight, PassInputs, PlayPattern
from xplain.scoring import ModelLoadError, ScoreResult, XPassScorer, find_default_model

from .test_explainer import random_features

SRC = Path(__file__).resolve().parents[1] / "src"
CFG = analytics.DEFAULT
NO_FLOOR = AnalyticsConfig(commentary_floor_success_max_xp=-1.0, commentary_floor_failure_min_xp=2.0)
WIDE = AnalyticsConfig(surprise_success_max_xp=0.6, surprise_failure_min_xp=0.7)  # many moments on the tiny model
KINDS = ("surprise_success", "surprise_failure")


@pytest.fixture(scope="module")
def tiny_model() -> CalibratedClassifierCV:
    """Small CalibratedClassifierCV over XGBClassifier, trained on random data with a fixed seed."""
    rng = np.random.default_rng(11)
    X = random_features(600, rng)
    logit = 2.0 - 0.03 * X["start_x"] - 1.0 * X["under_pressure"] - 1.5 * X["height_High Pass"]
    y = (rng.uniform(size=len(X)) < 1 / (1 + np.exp(-logit))).astype(int)
    base = XGBClassifier(n_estimators=8, max_depth=3, learning_rate=0.3, n_jobs=1, random_state=0)
    return CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X, y)


@pytest.fixture(scope="module")
def tiny_scorer(tiny_model, tmp_path_factory) -> XPassScorer:
    """The real XPassScorer class, loading the tiny model from a joblib file."""
    folder = tmp_path_factory.mktemp("tiny")
    (folder / "feature_list.json").write_text(json.dumps(FEATURE_ORDER), encoding="utf-8")
    joblib.dump(tiny_model, folder / "tiny.joblib")
    return XPassScorer(folder / "tiny.joblib", folder / "feature_list.json")


@pytest.fixture(scope="module")
def seed42_events() -> list:
    return generate_match(42)[0]


class FixedScorer:
    """Scores each pass with a chosen xP (looked up by start_x) and explains with the tiny boosters."""

    model_version = "fixed-xp"

    def __init__(self, boosters, xp_by_start_x: dict[float, float]):
        self.boosters = boosters
        self._xp = xp_by_start_x

    def score(self, inputs: Sequence[PassInputs]) -> list[ScoreResult]:
        return [ScoreResult(expected_probability=self._xp[p.start_x]) for p in inputs]


def make_event(seq: int, start_x: float, completed: bool, period: int = 1) -> PassEvent:
    return PassEvent(
        event_id=f"m1:{seq}", match_id="m1", seq=seq, t=10.0 * seq, period=period, team_id="brk", player_id="brk_p7",
        inputs=PassInputs(start_x=start_x, start_y=30.0, height=PassHeight.GROUND, body_part=BodyPart.FOOT,
                          play_pattern=PlayPattern.REGULAR, under_pressure=False),
        details=PassDetails(end_x=min(start_x + 10, 120.0), end_y=30.0, length=10.0, ball_speed_ms=14.0,
                            completed=completed),
    )


def run(tiny_scorer, plan: list[tuple], cfg: AnalyticsConfig = CFG, order=None) -> list[MomentBundle]:
    """Build events from (xP, completed[, period]) in seq order 1..n and run the pipeline with fixed xP values.

    Plans must keep periods non-decreasing (all period-1 steps before period-2 steps), like a real match.
    """
    events = [make_event(seq, 10.0 + seq, step[1], *step[2:]) for seq, step in enumerate(plan, start=1)]
    scorer = FixedScorer(tiny_scorer.boosters, {10.0 + seq: step[0] for seq, step in enumerate(plan, start=1)})
    if order is not None:
        events = [events[i] for i in order]
    return process_match(events, scorer, cfg)


def why(bundles: list[MomentBundle]) -> list[tuple[str, ...]]:
    return [b.qualified_by for b in bundles]


def two_halves_plan() -> list[tuple[float, bool, int]]:
    """Many moments of both kinds in both halves, arriving in increasing and decreasing significance."""
    n = 8
    # values stay strictly inside the outlier thresholds (failure xP >= 0.92, success xP <= 0.35)
    rising_fail = [(0.93 + 0.06 * i / (n - 1), False) for i in range(n)]        # significance up
    falling_fail = [(0.99 - 0.06 * i / (n - 1), False) for i in range(n)]       # significance down
    rising_success = [(0.34 - 0.32 * i / (n - 1), True) for i in range(n)]     # significance up
    falling_success = [(0.02 + 0.32 * i / (n - 1), True) for i in range(n)]    # significance down
    first = [step for pair in zip(rising_fail, falling_success) for step in pair]
    second = [step for pair in zip(falling_fail, rising_success) for step in pair]
    return [(*step, 1) for step in first] + [(*step, 2) for step in second]


# --- config -------------------------------------------------------------------------------------

def test_config_defaults_and_validation():
    assert (CFG.highlights_per_kind, CFG.commentary_floor_success_max_xp, CFG.commentary_floor_failure_min_xp) == (
        3, 0.25, 0.96)
    with pytest.raises(ValueError, match="commentary_floor_success_max_xp"):
        AnalyticsConfig(commentary_floor_success_max_xp=0.4)
    with pytest.raises(ValueError, match="commentary_floor_failure_min_xp"):
        AnalyticsConfig(commentary_floor_failure_min_xp=0.9)
    AnalyticsConfig(commentary_floor_success_max_xp=0.35, commentary_floor_failure_min_xp=0.92)  # equal is allowed
    assert not hasattr(CFG, "highlights_per_kind_full")


# --- ordering and pairing -----------------------------------------------------------------------

def test_bundles_in_seq_order_and_paired(tiny_scorer):
    plan = [(0.95, False), (0.5, True), (0.1, True), (0.99, False), (0.2, True)]
    bundles = run(tiny_scorer, plan, order=[3, 0, 4, 2, 1])  # events handed over shuffled
    assert [b.moment.seq for b in bundles] == [1, 3, 4, 5]   # seq 2 is not an outlier
    for b in bundles:
        assert b.moment.moment_id == b.explanation.moment_id
        assert b.moment.event_id == b.explanation.event_id


# --- commentary gate ----------------------------------------------------------------------------

def test_later_strong_moment_keeps_earlier_flag(tiny_scorer):
    cfg = AnalyticsConfig(highlights_per_kind=1)
    bundles = run(tiny_scorer, [(0.93, False), (0.99, False)], cfg)
    assert why(bundles) == [("ranking",), ("ranking", "threshold")]
    assert [m.seq for m in select_highlights([b.moment for b in bundles], 1)["surprise_failure"]] == [2]


def test_mild_moment_after_k_stronger_is_not_commentated(tiny_scorer):
    k = CFG.highlights_per_kind
    plan = ([(0.99 - 0.005 * i, False) for i in range(k)]   # K stronger failures fill the ranking
            + [(0.93, False),   # weaker than the K before it, below the floor: no commentary
               (0.30, True)])   # other kind: its own ranking is still empty
    bundles = run(tiny_scorer, plan)
    assert [b.qualifies_live for b in bundles] == [True] * k + [False, True]
    assert bundles[k].qualified_by == ()


def test_floor_only_and_both_reasons(tiny_scorer):
    cfg = AnalyticsConfig(highlights_per_kind=2)
    plan = [(0.99, False), (0.985, False),   # fill the failure ranking (K = 2)
            (0.97, False),                    # misses the ranking, clears the floor (>= 0.96): threshold only
            (0.95, False),                    # misses both
            (0.10, True)]                     # first success, under the floor (<= 0.25): both reasons
    bundles = run(tiny_scorer, plan, cfg)
    assert why(bundles) == [("ranking", "threshold"), ("ranking", "threshold"), ("threshold",), (), ("ranking", "threshold")]
    assert [b.qualifies_live for b in bundles] == [True, True, True, False, True]


def test_rankings_are_per_half(tiny_scorer):
    cfg = AnalyticsConfig(highlights_per_kind=1)
    second_half = [(0.93, False, 2), (0.30, True, 2)]
    alone = run(tiny_scorer, second_half, cfg)
    after_strong_first_half = run(tiny_scorer, [(0.99, False, 1), (0.05, True, 1)] + second_half, cfg)
    assert why(alone) == why(after_strong_first_half[2:]) == [("ranking",), ("ranking",)]
    ranks_alone = {i.bundle.moment.seq: i.rank for i in timeline(alone, 1)}
    ranks_after = {i.bundle.moment.seq - 2: i.rank for i in timeline(after_strong_first_half, 1)
                   if i.bundle.moment.period == 2}
    assert ranks_alone == ranks_after == {1: 1, 2: 1}


def test_failure_never_crowds_out_success(tiny_scorer):
    cfg = AnalyticsConfig(highlights_per_kind=1, commentary_floor_success_max_xp=-1.0,
                          commentary_floor_failure_min_xp=2.0)  # no floor: ranking only
    # failures with much higher significance (0.999) do not affect a mild success (significance 0.66)
    bundles = run(tiny_scorer, [(0.999, False), (0.998, False), (0.34, True)], cfg)
    assert why(bundles) == [("ranking",), (), ("ranking",)]


def test_flags_match_rule_on_earlier_moments_of_same_half(tiny_scorer):
    rng = np.random.default_rng(3)
    plan = ([(float(x), False) for x in rng.uniform(0.92, 1.0, 9)] + [(float(x), True) for x in rng.uniform(0, 0.35, 8)])
    first = [plan[i] for i in rng.permutation(len(plan))]
    second = [plan[i] for i in rng.permutation(len(plan))]
    bundles = run(tiny_scorer, [(*s, 1) for s in first] + [(*s, 2) for s in second])
    moments = [b.moment for b in bundles]
    assert len(bundles) == 2 * len(plan)
    for i, b in enumerate(bundles):
        same_half = [m for m in moments[:i] if m.period == b.moment.period]
        expected = (("ranking",) if qualifies_live(b.moment, same_half, CFG.highlights_per_kind) else ()) + (
            ("threshold",) if clears_floor(b.moment, CFG) else ())
        assert b.qualified_by == expected


def test_moment_bundle_validation(tiny_scorer):
    b = run(tiny_scorer, [(0.99, False)])[0]
    data = b.model_dump()
    MomentBundle.model_validate({**data, "qualifies_live": False, "qualified_by": ()})
    for bad in ({"qualifies_live": False}, {"qualified_by": ()},
                {"qualified_by": ("threshold", "ranking")}, {"qualified_by": ("ranking", "ranking")},
                {"qualified_by": ("hunch",)}):
        with pytest.raises(ValidationError):
            MomentBundle.model_validate({**data, **bad})
    assert MomentBundle.model_validate_json(b.model_dump_json()) == b


# --- real XPassScorer class on a generated match (tiny model) -----------------------------------

def test_generated_match_with_tiny_scorer(tiny_scorer, seed42_events):
    bundles = process_match(seed42_events, tiny_scorer, WIDE)
    assert bundles
    players, teams = names_from_meta(seed42_events)
    seqs = [b.moment.seq for b in bundles]
    assert seqs == sorted(seqs)
    for b in bundles:
        assert b.moment.player_name == players[b.moment.player_id]
        assert b.moment.team_name == teams[b.moment.team_id]
        assert additivity_error(b.explanation) <= 1e-4
    line = summary_line(tiny_scorer.model_version, 976, bundles)
    assert "threshold-only" in line and "h1" in line and "max additivity error" in line


def test_deterministic_json(tiny_scorer, seed42_events):
    runs = []
    for _ in range(2):
        bundles = process_match(seed42_events, tiny_scorer, WIDE)
        runs.append([b.model_dump_json() for b in bundles]
                    + [r.model_dump_json() for r in reports(bundles, 5)]
                    + [i.model_dump_json() for i in timeline(bundles, 5, 3000.0)])
    assert runs[0] == runs[1]


def test_no_llm_imports():
    code = ("import sys, xplain.pipeline; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('openai', 'azure')))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "[]"


# --- per-half reports ---------------------------------------------------------------------------

def assert_reports_valid(match_reports: list[Report], bundles: list[MomentBundle], per_kind: int) -> None:
    """Shape rules and the live-commentary guarantee shared by every report test."""
    assert [r.scope for r in match_reports] == ["first_half", "second_half"]
    for r, period in zip(match_reports, (1, 2)):
        assert set(r.highlights) == set(KINDS)
        for kind, top in r.highlights.items():
            assert len(top) <= per_kind
            assert all(b.moment.kind == kind and b.moment.period == period for b in top)
            keys = [(-b.moment.significance, b.moment.seq) for b in top]
            assert keys == sorted(keys)
            assert all(any(b is x for x in bundles) for b in top)  # same objects as the main list
            assert all(b.qualifies_live for b in top)              # reports reuse live commentary
        expected = select_half_highlights([b.moment for b in bundles], per_kind)[period]
        assert {k: [b.moment for b in v] for k, v in r.highlights.items()} == expected


@pytest.mark.parametrize("per_kind", [1, 2, 3, 5])
@pytest.mark.parametrize("cfg_floor", [CFG, NO_FLOOR], ids=["floor", "no_floor"])
def test_reports_property_synthetic(tiny_scorer, per_kind, cfg_floor):
    cfg = AnalyticsConfig(highlights_per_kind=per_kind,
                          commentary_floor_success_max_xp=cfg_floor.commentary_floor_success_max_xp,
                          commentary_floor_failure_min_xp=cfg_floor.commentary_floor_failure_min_xp)
    bundles = run(tiny_scorer, two_halves_plan(), cfg)
    assert len(bundles) == 32
    match_reports = reports(bundles, per_kind)
    assert_reports_valid(match_reports, bundles, per_kind)
    assert all(len(top) == per_kind for r in match_reports for top in r.highlights.values())


def test_reports_property_generated_match(tiny_scorer, seed42_events):
    bundles = process_match(seed42_events, tiny_scorer, WIDE)
    assert {b.moment.period for b in bundles} == {1, 2}
    assert_reports_valid(reports(bundles, WIDE.highlights_per_kind), bundles, WIDE.highlights_per_kind)


def test_empty_kinds_still_present(tiny_scorer):
    only_second_half_failures = run(tiny_scorer, [(0.95, False, 2), (0.99, False, 2)])
    first, second = reports(only_second_half_failures, 5)
    assert first.highlights == {"surprise_success": [], "surprise_failure": []}
    assert second.highlights["surprise_success"] == []
    assert [b.moment.seq for b in second.highlights["surprise_failure"]] == [2, 1]
    assert [r.highlights for r in reports([], 5)] == [{"surprise_success": [], "surprise_failure": []}] * 2


def test_report_contract_is_strict(tiny_scorer):
    bundles = run(tiny_scorer, [(0.95, False), (0.99, False)])
    with pytest.raises(ValidationError, match="keys"):
        Report(scope="first_half", highlights={"surprise_failure": bundles})
    with pytest.raises(ValidationError, match="not sorted"):
        Report(scope="first_half", highlights={"surprise_success": [], "surprise_failure": bundles})
    with pytest.raises(ValidationError, match="outside period"):
        Report(scope="second_half", highlights={"surprise_success": [], "surprise_failure": bundles[::-1]})
    with pytest.raises(ValidationError):
        Report(scope="full_time", highlights={"surprise_success": [], "surprise_failure": []})
    report = reports(bundles, 5)[0]
    assert Report.model_validate_json(report.model_dump_json()) == report


# --- timeline -----------------------------------------------------------------------------------

def ids(items) -> list[str]:
    return [i.bundle.moment.moment_id for i in items]


def assert_timeline_valid(items: list[TimelineItem], bundles: list[MomentBundle], per_kind: int,
                          as_of_t: float | None) -> None:
    """Rules every timeline must follow."""
    limit = float("inf") if as_of_t is None else as_of_t
    seqs = [i.bundle.moment.seq for i in items]
    assert seqs == sorted(seqs)
    assert all(i.bundle.qualifies_live and i.bundle.moment.t <= limit for i in items)
    assert ids(items) == [b.moment.moment_id for b in bundles if b.qualifies_live and b.moment.t <= limit]
    for kind in KINDS:
        for period in (1, 2):
            ranks = sorted(i.rank for i in items
                           if i.in_ranking and i.bundle.moment.kind == kind and i.bundle.moment.period == period)
            assert ranks == list(range(1, len(ranks) + 1)) and len(ranks) <= per_kind


def test_displaced_moment_stays_on_timeline(tiny_scorer):
    bundles = run(tiny_scorer, [(0.93, False), (0.95, False)], AnalyticsConfig(highlights_per_kind=1))
    end = timeline(bundles, 1)
    assert [(i.bundle.moment.seq, i.in_ranking, i.rank) for i in end] == [(1, False, None), (2, True, 1)]
    early = timeline(bundles, 1, as_of_t=10.0)
    assert [(i.bundle.moment.seq, i.in_ranking, i.rank) for i in early] == [(1, True, 1)]


def test_floor_only_item_never_in_ranking(tiny_scorer):
    cfg = AnalyticsConfig(highlights_per_kind=2)
    bundles = run(tiny_scorer, [(0.99, False), (0.985, False), (0.97, False), (0.98, False, 2)], cfg)
    assert bundles[2].qualified_by == ("threshold",)
    for as_of in (30.0, 35.0, 40.0, None):
        item = next(i for i in timeline(bundles, 2, as_of) if i.bundle.moment.seq == 3)
        assert (item.in_ranking, item.rank) == (False, None)


def test_never_qualified_is_never_on_timeline(tiny_scorer):
    bundles = run(tiny_scorer, [(0.99, False), (0.93, False), (0.3, True)], AnalyticsConfig(highlights_per_kind=1))
    assert why(bundles) == [("ranking", "threshold"), (), ("ranking",)]
    for as_of in (None, 15.0, 20.0, 25.0, 30.0):
        assert "moment:m1:2" not in ids(timeline(bundles, 1, as_of))


def test_nothing_after_as_of_t_and_ranks_unique(tiny_scorer):
    bundles = run(tiny_scorer, two_halves_plan(), AnalyticsConfig(highlights_per_kind=3))
    for as_of in (0.0, 5.0, 10.0, 15.0, 55.0, 160.0, 161.0, 170.0, 320.0, 1e9, None):
        assert_timeline_valid(timeline(bundles, 3, as_of), bundles, 3, as_of)
    assert timeline(bundles, 3, 5.0) == []


def test_none_equals_infinity(tiny_scorer):
    bundles = run(tiny_scorer, two_halves_plan(), AnalyticsConfig(highlights_per_kind=2))
    assert timeline(bundles, 2) == timeline(bundles, 2, float("inf"))


def test_end_of_match_ranking_equals_reports(tiny_scorer, seed42_events):
    bundles = process_match(seed42_events, tiny_scorer, WIDE)
    per_kind = WIDE.highlights_per_kind
    items = timeline(bundles, per_kind)
    assert_timeline_valid(items, bundles, per_kind, None)
    report_ids = {b.moment.moment_id for r in reports(bundles, per_kind) for top in r.highlights.values() for b in top}
    assert {i.bundle.moment.moment_id for i in items if i.in_ranking} == report_ids
    assert any(not i.in_ranking for i in items)  # some commentated moments were displaced


def test_as_of_minute_matches_moment_minute(tiny_scorer):
    # make_event puts seq n at t = 10 * n: seq 5 is t = 50 (minute 1), seq 6 is t = 60 (minute 2)
    bundles = run(tiny_scorer, [(0.95, False)] * 4 + [(0.2, True), (0.1, True), (0.3, True)])
    assert [b.moment.minute for b in bundles[4:]] == [1, 2, 2]
    for n in (0, 1, 2, 3):
        items = timeline(bundles, 5, minute_to_t(n))
        assert ids(items) == [b.moment.moment_id for b in bundles if b.qualifies_live and b.moment.minute <= n]
    assert "moment:m1:6" not in ids(timeline(bundles, 5, minute_to_t(1)))  # t = 60 exactly is minute 2
    assert "moment:m1:6" in ids(timeline(bundles, 5, minute_to_t(2)))


def test_timeline_item_rank_invariant(tiny_scorer):
    bundle = run(tiny_scorer, [(0.95, False)])[0]
    TimelineItem(bundle=bundle, in_ranking=True, rank=1)
    TimelineItem(bundle=bundle, in_ranking=False, rank=None)
    with pytest.raises(ValidationError, match="rank must be set"):
        TimelineItem(bundle=bundle, in_ranking=False, rank=2)
    with pytest.raises(ValidationError, match="rank must be set"):
        TimelineItem(bundle=bundle, in_ranking=True, rank=None)
    with pytest.raises(ValidationError):
        TimelineItem(bundle=bundle, in_ranking=True, rank=0)


# --- CLI ----------------------------------------------------------------------------------------

def test_ensure_match_file_uses_seed(tmp_path):
    path = tmp_path / "match.jsonl"
    assert ensure_match_file(path, seed=7) is True
    assert next(e for e in load_events(path) if isinstance(e, MatchMetaEvent)).seed == 7
    assert ensure_match_file(path, seed=8) is False  # existing file is kept
    assert next(e for e in load_events(path) if isinstance(e, MatchMetaEvent)).seed == 7


def test_cli_seed_out_and_timeline(tiny_scorer, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "find_default_model", lambda: ("unused", "unused"))
    monkeypatch.setattr(pipeline, "XPassScorer", lambda *_: tiny_scorer)
    match, out = tmp_path / "match_seed7.jsonl", tmp_path / "moments.jsonl"
    assert pipeline.main(["--file", str(match), "--seed", "7", "--out", str(out)]) == 0
    first = capsys.readouterr().out
    assert f"generated {match} with seed 7" in first and "qualified_by" in first
    assert next(e for e in load_events(match) if isinstance(e, MatchMetaEvent)).seed == 7
    assert reports_path(out) == tmp_path / "moments_reports.jsonl"
    lines = reports_path(out).read_text(encoding="utf-8").splitlines()
    assert [Report.model_validate_json(line).scope for line in lines] == ["first_half", "second_half"]
    for line in out.read_text(encoding="utf-8").splitlines():
        MomentBundle.model_validate_json(line)

    pipeline.main(["--file", str(match)])
    plain = capsys.readouterr().out
    assert "timeline at" not in plain
    pipeline.main(["--file", str(match), "--timeline", "--as-of-minute", "45"])
    with_timeline = capsys.readouterr().out
    assert with_timeline.startswith(plain)  # the flags only add sections after the default output
    assert "timeline at the end of the match" in with_timeline and "timeline at minute 45" in with_timeline


# --- real model (skipped when the joblib file is missing) ---------------------------------------

def test_real_model_seed42(seed42_events):
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc})")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path}")
    bundles = process_match(seed42_events, XPassScorer(model_path, feature_list_path))
    assert len(bundles) == 42
    assert Counter(b.moment.kind for b in bundles) == {"surprise_success": 21, "surprise_failure": 21}
    for b in bundles:
        assert MomentBundle.model_validate_json(b.model_dump_json()) == b
        assert additivity_error(b.explanation) <= 1e-4
    per_kind = CFG.highlights_per_kind
    match_reports = reports(bundles, per_kind)
    assert_reports_valid(match_reports, bundles, per_kind)
    items = timeline(bundles, per_kind)
    assert_timeline_valid(items, bundles, per_kind, None)
    report_ids = {b.moment.moment_id for r in match_reports for top in r.highlights.values() for b in top}
    assert {i.bundle.moment.moment_id for i in items if i.in_ranking} == report_ids
