import asyncio
import json
import math

import pytest
from pydantic import ValidationError

from xplain import bundle as bundle_mod
from xplain import clock
from xplain.bundle import (
    MatchBundle,
    build_bundle,
    clock_track,
    main,
    narrate_match,
    offline_outcomes,
    stored_lines,
    to_json,
)
from xplain.narration import difficulty_tier
from xplain.narrator import Narrator
from xplain.pipeline import analyse_match, reports, timeline, write_jsonl
from xplain.contracts import PassMark
from xplain.schema import MatchMetaEvent, PassEvent, PlayPattern, ShotEvent

from .test_pipeline import KINDS, WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_workflow import BAD, GOOD, FakeAgent


def narrated(events, scorer, answer, stored=None):
    agent = FakeAgent(answer)
    outcomes = asyncio.run(narrate_match(events, scorer, Narrator(agent, mode="hints"), stored, cfg=WIDE))
    return outcomes, agent


@pytest.fixture(scope="module")
def offline(seed42_events, tiny_scorer):  # noqa: F811
    return build_bundle(seed42_events, tiny_scorer, None, "hints", WIDE)


@pytest.fixture(scope="module")
def online(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    return build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)


def test_json_round_trips(offline, online):
    for b in (offline, online):
        assert MatchBundle.model_validate_json(to_json(b)) == b


def test_bundle_holds_every_pass_and_outlier(seed42_events, tiny_scorer, offline):  # noqa: F811
    bundles, marks = analyse_match(seed42_events, tiny_scorer, WIDE)
    assert [PassMark(**m.model_dump(exclude={"display"})) for m in offline.marks] == marks
    assert [m.bundle for m in offline.moments] == bundles
    assert offline.meta.schema_version == "5" and offline.meta.model_version == tiny_scorer.model_version
    meta = next(e for e in seed42_events if isinstance(e, MatchMetaEvent))
    assert offline.meta.generator_version == meta.generator_version and len(offline.meta.players) == 22


def test_minute_labels_come_from_match_clock(seed42_events, offline):  # noqa: F811
    start = clock.second_half_start(seed42_events)
    for m in offline.marks:
        assert m.minute_label == clock.match_clock(m.t, m.period, start).label
    for e in offline.moments:
        m = e.bundle.moment
        assert m.minute_label == clock.match_clock(m.t, m.period, start).label


def test_clock_ticks_cover_each_period_by_minute(seed42_events, offline):  # noqa: F811
    c = offline.clock
    start = clock.second_half_start(seed42_events)
    assert c.second_half_start == start and c.first_event_t == min(e.t for e in seed42_events)
    assert c.last_event_t == max(e.t for e in seed42_events)
    ts = [k.t for k in c.ticks]
    assert ts == sorted(ts) and ts[0] == 0.0 and start in ts
    for k in c.ticks:
        period = 2 if k.t >= start else 1
        assert k.minute_label == clock.match_clock(k.t, period, start).label
    labels = [k.minute_label for k in c.ticks]
    assert labels[:2] == ["1", "2"] and "45+1" in labels and "46" in labels and "90" in labels
    assert len(labels) == len(set(labels))   # one tick per match minute
    # every event's label is the label of the last tick at or before it
    for e in seed42_events:
        shown = [k for k in c.ticks if k.t <= e.t][-1].minute_label
        assert shown == clock.match_clock(e.t, e.period, start).label


def test_first_half_only_match_has_no_second_half(seed42_events):  # noqa: F811
    c = clock_track([e for e in seed42_events if e.period == 1])
    assert c.second_half_start is None and c.ticks[-1].minute_label.startswith("45")


def test_report_ids_exist_in_moments(offline, online):
    for b in (offline, online):
        ids = {m.bundle.moment.moment_id for m in b.moments}
        assert [r.scope for r in b.reports] == ["first_half", "second_half"]
        assert all(i in ids for r in b.reports for v in r.highlights.values() for i in v)
        assert sum(len(v) for r in b.reports for v in r.highlights.values()) > 0


def test_lines_only_on_commentated_moments(offline, online):
    for e in offline.moments:
        assert e.line is None
        assert e.narration.status == ("not_run" if e.bundle.qualifies_live else "not_commentated")
    assert any(not e.bundle.qualifies_live for e in online.moments)
    for e in online.moments:
        if e.bundle.qualifies_live:
            assert e.narration.status == "accepted" and e.line is not None and e.narration.attempts == 1
        else:
            assert e.narration.status == "not_commentated" and e.line is None


def test_rejected_moments_keep_reasons(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = narrated(seed42_events, tiny_scorer, lambda i, f: BAD(f))
    b = build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)
    live = [e for e in b.moments if e.bundle.qualifies_live]
    assert live and all(e.narration.status == "rejected" and e.line is None and e.narration.reasons for e in live)


def test_players_map_covers_all_22(seed42_events, offline):  # noqa: F811
    meta = next(e for e in seed42_events if isinstance(e, MatchMetaEvent))
    assert list(offline.players) == [p.player_id for p in meta.players]
    passed = {m.player_id for m in offline.marks}
    for pid, s in offline.players.items():
        assert (s is None) == (pid not in passed)
        if s is not None:
            assert s.passes == sum(m.player_id == pid for m in offline.marks)


def test_score_counts_goals_in_order(seed42_events, tiny_scorer):  # noqa: F811
    meta = next(e for e in seed42_events if isinstance(e, MatchMetaEvent))
    shots = [e for e in seed42_events if isinstance(e, ShotEvent)]
    assert len(shots) >= 3
    picked = {shots[0].event_id, shots[1].event_id, shots[-1].event_id}
    events = [e.model_copy(update={"goal": True, "on_target": True}) if e.event_id in picked else
              e.model_copy(update={"goal": False}) if isinstance(e, ShotEvent) else e for e in seed42_events]
    b = build_bundle(events, tiny_scorer, None, "hints", WIDE)
    start = clock.second_half_start(events)
    assert (b.score[0].home, b.score[0].away, b.score[0].goal) == (0, 0, None)
    goals = sorted((e for e in events if e.event_id in picked), key=lambda e: e.seq)
    assert len(b.score) == 4
    home = away = 0
    for entry, shot in zip(b.score[1:], goals, strict=True):
        home += shot.team_id == meta.home_team_id
        away += shot.team_id != meta.home_team_id
        assert (entry.home, entry.away, entry.goal.team_id, entry.t) == (home, away, shot.team_id, shot.t)
        assert entry.minute_label == clock.match_clock(shot.t, shot.period, start).label


def test_line_reuse_skips_the_model(seed42_events, tiny_scorer, online, tmp_path):  # noqa: F811
    out = tmp_path / "bundle.json"
    out.write_text(to_json(online), encoding="utf-8")
    stored = stored_lines(out, online.meta.match_id, online.meta.model_version)
    live = [e for e in online.moments if e.bundle.qualifies_live]
    assert set(stored) == {e.bundle.moment.moment_id for e in live}
    outcomes, agent = narrated(seed42_events, tiny_scorer, lambda i, f: BAD(f), stored)
    assert agent.calls == []
    again = build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)
    assert to_json(again) == to_json(online)
    # partial store: only the missing moments reach the model
    first = live[0].bundle.moment.moment_id
    _, agent = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f), {first: stored[first]})
    assert len(agent.calls) == len(live) - 1


def test_reuse_needs_same_match_and_model(online, tmp_path):
    out = tmp_path / "bundle.json"
    out.write_text(to_json(online), encoding="utf-8")
    assert stored_lines(out, "other-match", online.meta.model_version) == {}
    assert stored_lines(out, online.meta.match_id, "other-model") == {}
    assert stored_lines(tmp_path / "missing.json", online.meta.match_id, online.meta.model_version) == {}
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert stored_lines(tmp_path / "bad.json", online.meta.match_id, online.meta.model_version) == {}


def test_offline_reuse_runs_the_final_check(seed42_events, tiny_scorer, online, tmp_path):  # noqa: F811
    out = tmp_path / "bundle.json"
    out.write_text(to_json(online), encoding="utf-8")
    stored = stored_lines(out, online.meta.match_id, online.meta.model_version)
    b = build_bundle(seed42_events, tiny_scorer, offline_outcomes(seed42_events, tiny_scorer, stored, "hints", WIDE),
                     "hints", WIDE)
    assert to_json(b) == to_json(online)
    # a stored line that fails the final check is dropped
    first = next(iter(stored))
    broken = stored[first].model_copy(update={"line": stored[first].line.model_copy(update={"line_en": "A 95% pass."})})
    outcomes = offline_outcomes(seed42_events, tiny_scorer, {first: broken}, "hints", WIDE)
    assert len(outcomes) == 1 and outcomes[0].line is None and outcomes[0].status == "rejected"
    assert any(r.startswith("final check: no_numbers") for r in outcomes[0].reasons)


def test_same_input_gives_identical_json(seed42_events, tiny_scorer, offline):  # noqa: F811
    assert to_json(build_bundle(seed42_events, tiny_scorer, None, "hints", WIDE)) == to_json(offline)


def test_no_seconds_labels_in_bundle(offline):
    text = to_json(offline)
    assert '"minute_label":"' in text
    labels = {k.minute_label for k in offline.clock.ticks}
    assert all(":" not in label for label in labels)


def test_cli_no_llm_writes_bundle_and_reuses_lines(seed42_events, tiny_scorer, online, tmp_path, monkeypatch,  # noqa: F811
                                                   capsys):
    match = tmp_path / "match.jsonl"
    write_jsonl(match, seed42_events)
    out = tmp_path / "bundle.json"
    monkeypatch.setattr(bundle_mod, "_load_scorer", lambda: tiny_scorer)
    assert main(["--file", str(match), "--out", str(out), "--no-llm", "--mode", "hints"]) == 0
    line = capsys.readouterr().out.strip()
    b = MatchBundle.model_validate_json(out.read_text(encoding="utf-8"))
    n_live = sum(e.bundle.qualifies_live for e in b.moments)
    assert line.startswith(f"{len(b.marks)} passes, {len(b.moments)} moments, {n_live} commentated: ")
    assert f"not_run {n_live}; goals {len(b.score) - 1}, final score" in line and line.endswith("KB)")
    # stored lines are picked up from --out on the next run, and ignored with --fresh
    out.write_text(to_json(online), encoding="utf-8")
    n_stored = sum(e.line is not None for e in online.moments)
    assert main(["--file", str(match), "--out", str(out), "--no-llm", "--mode", "hints"]) == 0
    assert f"reusing {n_stored} stored line(s)" in capsys.readouterr().err
    out.write_text(to_json(online), encoding="utf-8")
    assert main(["--file", str(match), "--out", str(out), "--no-llm", "--mode", "hints", "--fresh"]) == 0
    assert "reusing" not in capsys.readouterr().err


# --- ranking snapshots (schema 4: scopes all / 1 / 2) and tiers ----------------------------------

SCOPES = ("all", "1", "2")


def in_scope(m, scope):
    return scope == "all" or str(m.period) == scope


def test_two_snapshots_per_commentated_moment_in_order(offline):
    live = sorted((e.bundle.moment for e in offline.moments if e.bundle.qualifies_live), key=lambda m: m.seq)
    assert len(live) >= 4
    want = [(m.t, scope, m.kind) for m in live for scope in ("all", str(m.period))]
    assert [(s.t, s.scope, s.kind) for s in offline.rankings] == want


def test_snapshot_ranks_equal_a_fresh_sort_of_the_eligible_moments(offline):
    moments = [e.bundle.moment for e in offline.moments]
    seen = set()
    for snap in offline.rankings:
        eligible = [m for m in moments if m.kind == snap.kind and m.t <= snap.t and in_scope(m, snap.scope)]
        fresh = sorted(eligible, key=lambda m: (-abs(m.delta), m.seq))[:WIDE.highlights_per_kind]
        assert [e.moment_id for e in snap.entries] == [m.moment_id for m in fresh]
        assert [e.rank for e in snap.entries] == list(range(1, len(fresh) + 1))
        seen.add(snap.scope)
    assert seen == set(SCOPES)


def test_half_scopes_equal_the_pipeline_timeline(offline):
    bundles = [e.bundle for e in offline.moments]
    for snap in (s for s in offline.rankings if s.scope != "all"):
        expected = sorted(((i.rank, i.bundle.moment.moment_id) for i in timeline(bundles, WIDE.highlights_per_kind, snap.t)
                           if i.in_ranking and i.bundle.moment.kind == snap.kind
                           and str(i.bundle.moment.period) == snap.scope))
        assert [(e.rank, e.moment_id) for e in snap.entries] == expected


def test_previous_rank_comes_from_the_previous_snapshot_of_the_same_scope_and_kind(offline):
    last: dict = {}
    seen_new = seen_carried = False
    for snap in offline.rankings:
        before = last.get((snap.scope, snap.kind), {})
        for e in snap.entries:
            assert e.previous_rank == before.get(e.moment_id)
            seen_new |= e.previous_rank is None
            seen_carried |= e.previous_rank is not None
        last[(snap.scope, snap.kind)] = {e.moment_id: e.rank for e in snap.entries}
    assert seen_new and seen_carried
    firsts = {}
    for snap in offline.rankings:
        firsts.setdefault((snap.scope, snap.kind), snap)
    for snap in firsts.values():
        assert all(e.previous_rank is None for e in snap.entries)


def test_first_half_ranking_is_frozen_after_half_time(seed42_events, tiny_scorer, offline):  # noqa: F811
    start = clock.second_half_start(seed42_events)
    final_h1 = reports([e.bundle for e in offline.moments], WIDE.highlights_per_kind)[0]
    for kind in KINDS:
        h1 = [s for s in offline.rankings if s.scope == "1" and s.kind == kind]
        assert h1 and all(s.t < start for s in h1)
        assert [e.moment_id for e in h1[-1].entries] == [b.moment.moment_id for b in final_h1.highlights[kind]]
    assert all(s.t >= start for s in offline.rankings if s.scope == "2")


def test_all_scope_holds_both_halves_and_only_commentated_moments(offline):
    by_id = {e.bundle.moment.moment_id: e.bundle for e in offline.moments}
    for snap in offline.rankings:
        assert all(by_id[e.moment_id].qualifies_live for e in snap.entries)
        assert all(in_scope(by_id[e.moment_id].moment, snap.scope) for e in snap.entries)
    halves = {by_id[e.moment_id].moment.period for s in offline.rankings if s.scope == "all" for e in s.entries}
    assert halves == {1, 2}


def test_tier_words_come_from_difficulty_tier(offline):
    for e in offline.moments:
        t = difficulty_tier(e.bundle.moment)
        assert (e.tier.family, e.tier.en, e.tier.es, e.tier.el) == (t.family, t.en, t.es, t.el)


def test_bad_snapshots_are_rejected(offline):
    data = offline.model_dump(mode="json")
    snap = next(s for s in data["rankings"] if len(s["entries"]) >= 2)
    snap["entries"][0]["rank"] = 2
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)
    data = offline.model_dump(mode="json")
    data["rankings"][0]["entries"][0]["moment_id"] = "moment:nope"
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)
    data = offline.model_dump(mode="json")
    other = next(s for s in data["rankings"] if s["kind"] != data["rankings"][0]["kind"])
    data["rankings"][0]["entries"] = other["entries"]
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)
    data = offline.model_dump(mode="json")
    h2 = next(s for s in data["rankings"] if s["scope"] == "2" and s["entries"])
    data["rankings"][0]["scope"] = "1"
    data["rankings"][0]["kind"] = h2["kind"]
    data["rankings"][0]["entries"] = h2["entries"][:1]
    with pytest.raises(ValidationError):   # a second-half moment in a first-half table
        MatchBundle.model_validate(data)
    data = offline.model_dump(mode="json")
    quiet = next(m["bundle"]["moment"] for m in data["moments"] if not m["bundle"]["qualifies_live"])
    snap = next(s for s in data["rankings"] if s["scope"] == "all" and s["kind"] == quiet["kind"]
                and s["t"] >= quiet["t"])
    snap["entries"][0]["moment_id"] = quiet["moment_id"]
    with pytest.raises(ValidationError):   # a moment that was never commentated
        MatchBundle.model_validate(data)


def untraced(stored: dict) -> dict:
    """Stored lines as an old-schema file gives them: no trace."""
    return {k: v.model_copy(update={"trace": []}) for k, v in stored.items()}


def as_schema_1(b: MatchBundle) -> str:
    """The bundle as the schema "1" file it would have been: no rankings, no tier."""
    data = b.model_dump(mode="json")
    data["meta"]["schema_version"] = "1"
    del data["rankings"], data["trace"]
    for m in data["moments"]:
        del m["tier"]
    return json.dumps(data)


def test_line_reuse_reads_a_schema_1_file(online, tmp_path):
    new, old = tmp_path / "v2.json", tmp_path / "v1.json"
    new.write_text(to_json(online), encoding="utf-8")
    old.write_text(as_schema_1(online), encoding="utf-8")
    from_v2 = stored_lines(new, online.meta.match_id, online.meta.model_version)
    assert from_v2 and stored_lines(old, online.meta.match_id, online.meta.model_version) == untraced(from_v2)
    assert stored_lines(old, "other-match", online.meta.model_version) == {}
    assert stored_lines(old, online.meta.match_id, "other-model") == {}
    data = json.loads(as_schema_1(online))
    data["meta"]["schema_version"] = "0"
    old.write_text(json.dumps(data), encoding="utf-8")
    assert stored_lines(old, online.meta.match_id, online.meta.model_version) == {}


def as_schema_2(b: MatchBundle) -> str:
    """The bundle as the schema "2" file it would have been: no display geometry, no attacks_right."""
    data = b.model_dump(mode="json")
    data["meta"]["schema_version"] = "2"
    del data["meta"]["attacks_right"], data["trace"]
    for m in data["marks"]:
        del m["display"]
    return json.dumps(data)


def test_line_reuse_reads_a_schema_2_file(online, tmp_path):
    new, old = tmp_path / "v3.json", tmp_path / "v2.json"
    new.write_text(to_json(online), encoding="utf-8")
    old.write_text(as_schema_2(online), encoding="utf-8")
    from_v3 = stored_lines(new, online.meta.match_id, online.meta.model_version)
    assert from_v3 and stored_lines(old, online.meta.match_id, online.meta.model_version) == untraced(from_v3)


# --- PVA and the display frame (schema 3) --------------------------------------------------------

def test_pva_is_outcome_minus_xp(offline):
    for e in offline.moments:
        m = e.bundle.moment
        assert m.delta == pytest.approx((1.0 if m.completed else 0.0) - m.xp, abs=1e-12)
    for m in offline.marks:
        assert m.delta == pytest.approx((1.0 if m.completed else 0.0) - m.xp, abs=1e-12)


RESTARTS = {PlayPattern.CORNER, PlayPattern.GOAL_KICK, PlayPattern.THROW_IN, PlayPattern.FREE_KICK,
            PlayPattern.KICK_OFF}
LINE_UP_TOLERANCE = 0.05   # metres; on seed42 every open-play turnover lines up to 0.00


def turnover_gaps(seed42_events, offline):  # noqa: F811
    """(display gap, stored gap) from a failed pass's end to the next pass's start, at each change of
    team within a half where the next pass is not a restart."""
    pattern = {e.event_id: e.inputs.play_pattern for e in seed42_events if isinstance(e, PassEvent)}
    out = []
    for p, n in zip(offline.marks, offline.marks[1:], strict=False):
        if p.period != n.period or p.team_id == n.team_id or p.completed or pattern[n.event_id] in RESTARTS:
            continue
        d = math.dist((p.display.end_x, p.display.end_y), (n.display.start_x, n.display.start_y))
        raw = math.dist((p.pass_geometry.end_x, p.pass_geometry.end_y),
                        (n.pass_geometry.start_x, n.pass_geometry.start_y))
        out.append((d, raw))
    return out


def test_display_lines_up_at_turnovers(seed42_events, offline):  # noqa: F811
    gaps = turnover_gaps(seed42_events, offline)
    assert len(gaps) >= 50
    assert max(d for d, _ in gaps) <= LINE_UP_TOLERANCE
    # without the display frame the same pairs are far apart: the test has teeth
    assert sorted(r for _, r in gaps)[len(gaps) // 2] > 20


def test_display_rotates_the_right_team_in_each_half(offline):
    home, away = offline.meta.home.team_id, offline.meta.away.team_id
    assert offline.meta.attacks_right == {"1": home, "2": away}
    seen = set()
    for m in offline.marks:
        g, d = m.pass_geometry, m.display
        rotated = (m.team_id == home) == (m.period == 2)
        want = ((120 - g.start_x, 80 - g.start_y, 120 - g.end_x, 80 - g.end_y) if rotated
                else (g.start_x, g.start_y, g.end_x, g.end_y))
        assert (d.start_x, d.start_y, d.end_x, d.end_y) == pytest.approx(want)
        seen.add((m.team_id, m.period, rotated))
    assert seen == {(home, 1, False), (away, 1, True), (home, 2, True), (away, 2, False)}


def test_display_leaves_model_inputs_unchanged(seed42_events, tiny_scorer, offline):  # noqa: F811
    before = [e.inputs for e in seed42_events if isinstance(e, PassEvent)]
    bundles, marks = analyse_match(seed42_events, tiny_scorer, WIDE)
    again = build_bundle(seed42_events, tiny_scorer, None, "hints", WIDE)
    assert [e.inputs for e in seed42_events if isinstance(e, PassEvent)] == before
    assert [m.pass_geometry for m in again.marks] == [m.pass_geometry for m in marks]
    assert [(e.bundle.moment.evidence, e.bundle.explanation) for e in again.moments] == \
        [(b.moment.evidence, b.explanation) for b in bundles]


def test_wrong_display_geometry_is_rejected(offline):
    data = offline.model_dump(mode="json")
    data["marks"][0]["display"]["end_x"] += 1.0
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)
    data = offline.model_dump(mode="json")
    data["meta"]["attacks_right"] = {"1": data["meta"]["away"]["team_id"], "2": data["meta"]["home"]["team_id"]}
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)


def test_line_reuse_reads_a_schema_3_file(online, tmp_path):
    new, old = tmp_path / "v4.json", tmp_path / "v3.json"
    new.write_text(to_json(online), encoding="utf-8")
    data = online.model_dump(mode="json")
    data["meta"]["schema_version"] = "3"
    del data["trace"]
    for snap in data["rankings"]:
        snap["period"] = 1
        del snap["scope"]
    old.write_text(json.dumps(data), encoding="utf-8")
    from_v4 = stored_lines(new, online.meta.match_id, online.meta.model_version)
    assert from_v4 and stored_lines(old, online.meta.match_id, online.meta.model_version) == untraced(from_v4)


# --- trace (schema 5) ----------------------------------------------------------------------------

def test_trace_holds_every_commentated_moment(online):
    live = {e.bundle.moment.moment_id for e in online.moments if e.bundle.qualifies_live}
    assert online.meta.schema_version == "5" and set(online.trace.moments) == live
    assert online.trace.recaps == {}
    for steps in online.trace.moments.values():
        assert [s.agent for s in steps[:2]] == ["analyst", "explainer"] and steps[-1].final


def test_trace_round_trips_and_names_only_commentated_moments(online, offline):
    assert MatchBundle.model_validate_json(to_json(online)).trace == online.trace
    assert offline.trace.moments == {}   # --no-llm without stored lines: nothing ran
    data = online.model_dump(mode="json")
    quiet = next(m["bundle"]["moment"]["moment_id"] for m in data["moments"] if not m["bundle"]["qualifies_live"])
    data["trace"]["moments"][quiet] = []
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)


def test_a_schema_4_bundle_still_loads(online):
    data = online.model_dump(mode="json")
    data["meta"]["schema_version"] = "4"
    del data["trace"]
    old = MatchBundle.model_validate_json(json.dumps(data))
    assert old.meta.schema_version == "4" and old.trace.moments == {} and old.moments == online.moments
    data["trace"] = online.model_dump(mode="json")["trace"]
    with pytest.raises(ValidationError):   # a trace is schema 5 only
        MatchBundle.model_validate(data)


def test_reused_lines_keep_their_narrator_steps(online, seed42_events, tiny_scorer, tmp_path):  # noqa: F811
    path = tmp_path / "b.json"
    path.write_text(to_json(online), encoding="utf-8")
    stored = stored_lines(path, online.meta.match_id, online.meta.model_version)
    moment_id, result = next(iter(stored.items()))
    assert result.trace == [s for s in online.trace.moments[moment_id]
                            if s.agent in ("narrator", "verifier") and not s.final]
    outcomes, agent = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f), stored)
    again = build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)
    assert again.trace.moments[moment_id] == online.trace.moments[moment_id]
    offline_again = build_bundle(seed42_events, tiny_scorer,
                                 offline_outcomes(seed42_events, tiny_scorer, stored, "hints", WIDE), "hints", WIDE)
    assert offline_again.trace.moments[moment_id] == online.trace.moments[moment_id]
