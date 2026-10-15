import asyncio
import json

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
from xplain.pipeline import analyse_match, timeline, write_jsonl
from xplain.schema import MatchMetaEvent, ShotEvent

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
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
    assert offline.marks == marks
    assert [m.bundle for m in offline.moments] == bundles
    assert offline.meta.schema_version == "2" and offline.meta.model_version == tiny_scorer.model_version
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


# --- ranking snapshots and tiers (schema 2) ------------------------------------------------------

def test_one_snapshot_per_commentated_moment_in_order(offline):
    live = sorted((e.bundle.moment for e in offline.moments if e.bundle.qualifies_live), key=lambda m: m.seq)
    assert len(live) >= 4
    assert [(s.t, s.kind, s.period) for s in offline.rankings] == [(m.t, m.kind, m.period) for m in live]


def test_snapshot_ranks_match_the_timeline(offline):
    bundles = [e.bundle for e in offline.moments]
    for snap in offline.rankings:
        expected = sorted(((i.rank, i.bundle.moment.moment_id) for i in timeline(bundles, WIDE.highlights_per_kind, snap.t)
                           if i.in_ranking and i.bundle.moment.kind == snap.kind
                           and i.bundle.moment.period == snap.period))
        assert [(e.rank, e.moment_id) for e in snap.entries] == expected
        assert len(snap.entries) <= WIDE.highlights_per_kind


def test_previous_rank_comes_from_the_previous_snapshot_of_the_same_kind_and_half(offline):
    last: dict = {}
    seen_new = seen_carried = False
    for snap in offline.rankings:
        before = last.get((snap.kind, snap.period), {})
        for e in snap.entries:
            assert e.previous_rank == before.get(e.moment_id)
            seen_new |= e.previous_rank is None
            seen_carried |= e.previous_rank is not None
        last[(snap.kind, snap.period)] = {e.moment_id: e.rank for e in snap.entries}
    assert seen_new and seen_carried


def test_rankings_reset_at_half_time(offline):
    period = {e.bundle.moment.moment_id: e.bundle.moment.period for e in offline.moments}
    for snap in offline.rankings:
        assert all(period[e.moment_id] == snap.period for e in snap.entries)
    firsts = {}
    for snap in offline.rankings:
        firsts.setdefault((snap.kind, snap.period), snap)
    for snap in firsts.values():
        assert all(e.previous_rank is None for e in snap.entries)


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


def as_schema_1(b: MatchBundle) -> str:
    """The bundle as the schema "1" file it would have been: no rankings, no tier."""
    data = b.model_dump(mode="json")
    data["meta"]["schema_version"] = "1"
    del data["rankings"]
    for m in data["moments"]:
        del m["tier"]
    return json.dumps(data)


def test_line_reuse_reads_a_schema_1_file(online, tmp_path):
    new, old = tmp_path / "v2.json", tmp_path / "v1.json"
    new.write_text(to_json(online), encoding="utf-8")
    old.write_text(as_schema_1(online), encoding="utf-8")
    from_v2 = stored_lines(new, online.meta.match_id, online.meta.model_version)
    assert from_v2 and stored_lines(old, online.meta.match_id, online.meta.model_version) == from_v2
    assert stored_lines(old, "other-match", online.meta.model_version) == {}
    assert stored_lines(old, online.meta.match_id, "other-model") == {}
    data = json.loads(as_schema_1(online))
    data["meta"]["schema_version"] = "0"
    old.write_text(json.dumps(data), encoding="utf-8")
    assert stored_lines(old, online.meta.match_id, online.meta.model_version) == {}
