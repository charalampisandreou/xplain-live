import json
import re
from collections import Counter
from pathlib import Path

import pytest
from pydantic import TypeAdapter

import xplain.generator as generator
from xplain.generator import generate_match, write_match
from xplain.schema import Event

EVENT_ADAPTER = TypeAdapter(Event)


@pytest.fixture(scope="module")
def match42(tmp_path_factory):
    out = tmp_path_factory.mktemp("gen") / "match_seed42.jsonl"
    events_path, truth_path = write_match(42, out)
    lines = events_path.read_text(encoding="utf-8").splitlines()
    events = [EVENT_ADAPTER.validate_json(line) for line in lines]
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    return events_path, truth_path, events, truth


def test_same_seed_is_byte_identical(tmp_path, match42):
    events_path, truth_path, _, _ = match42
    again, again_truth = write_match(42, tmp_path / "again.jsonl")
    assert again.read_bytes() == events_path.read_bytes()
    assert again_truth.read_bytes() == truth_path.read_bytes()


def test_different_seeds_differ(tmp_path, match42):
    events_path, _, _, _ = match42
    other, _ = write_match(7, tmp_path / "seed7.jsonl")
    assert other.read_bytes() != events_path.read_bytes()


def test_truth_file_location_and_content(match42):
    events_path, truth_path, _, truth = match42
    assert truth_path.name == "match_seed42.truth.json"
    assert truth["seed"] == 42 and truth["generator_version"] == "0.1.1"
    assert truth["config"]["intercept"] == pytest.approx(9.19253)
    assert len(truth["players"]) == 22
    assert all(isinstance(p["pass_skill"], float) for p in truth["players"])


def test_every_line_parses_and_is_ordered(match42):
    _, _, events, _ = match42
    assert [e.seq for e in events] == list(range(len(events)))
    assert len({e.event_id for e in events}) == len(events)
    assert all(e.match_id == "m42" for e in events)
    times = [e.t for e in events]
    assert times == sorted(times)
    last_p1 = max(i for i, e in enumerate(events) if e.period == 1)
    first_p2 = min(i for i, e in enumerate(events) if e.period == 2)
    assert last_p1 < first_p2


def test_first_event_is_match_meta(match42):
    _, _, events, _ = match42
    meta = events[0]
    assert meta.type == "match_meta" and meta.seq == 0 and meta.t == 0.0 and meta.period == 1
    assert meta.team_id == meta.home_team_id and meta.seed == 42
    assert sum(e.type == "match_meta" for e in events) == 1


def test_all_event_types_present(match42):
    _, _, events, _ = match42
    types = Counter(e.type for e in events)
    for kind in ("match_meta", "pass", "shot", "tackle", "possession_change", "pressure", "player_load"):
        assert types[kind] > 0, kind


def test_teams_and_player_membership(match42):
    _, _, events, _ = match42
    meta = events[0]
    team_of = {p.player_id: p.team_id for p in meta.players}
    by_team = Counter(team_of.values())
    assert by_team == {meta.home_team_id: 11, meta.away_team_id: 11}
    assert len({p.name for p in meta.players}) == 22
    for team in (meta.home_team_id, meta.away_team_id):
        shape = Counter(p.position for p in meta.players if p.team_id == team)
        assert shape == {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}
    for e in events:
        if e.type in ("pass", "shot", "tackle", "pressure", "player_load"):
            assert team_of[e.player_id] == e.team_id, e.event_id
        if e.type == "pass" and e.details.recipient_id is not None:
            assert team_of[e.details.recipient_id] == e.team_id
            assert e.details.recipient_id != e.player_id
        if e.type == "pressure":
            assert team_of[e.pressed_player_id] != e.team_id


def test_pass_volume_and_completion_rate(match42):
    _, _, events, _ = match42
    passes = [e for e in events if e.type == "pass"]
    assert 500 <= len(passes) <= 1300
    rate = sum(p.details.completed for p in passes) / len(passes)
    assert 0.70 <= rate <= 0.90


def test_completed_passes_have_recipient(match42):
    _, _, events, _ = match42
    for p in (e for e in events if e.type == "pass"):
        if p.details.completed:
            assert p.details.recipient_id is not None


def test_pressure_precedes_pressured_pass(match42):
    _, _, events, _ = match42
    for i, e in enumerate(events):
        if e.type == "pass" and e.inputs.under_pressure:
            prev = events[i - 1]
            assert prev.type == "pressure" and prev.pressed_player_id == e.player_id
            assert e.t - prev.t <= 2.0 + 1e-9


def test_player_load_is_cumulative(match42):
    _, _, events, _ = match42
    last: dict[str, tuple[float, float, int]] = {}
    for e in (e for e in events if e.type == "player_load"):
        if e.player_id in last:
            d, v, s = last[e.player_id]
            assert e.distance_m >= d and e.top_speed_ms >= v and e.sprint_count >= s
        assert e.top_speed_ms <= 10.5
        last[e.player_id] = (e.distance_m, e.top_speed_ms, e.sprint_count)
    assert len(last) == 22
    # full-time totals: outfield players cover far more ground than goalkeepers
    meta = events[0]
    gks = {p.player_id for p in meta.players if p.position == "GK"}
    assert all(3000 < last[g][0] < 6500 for g in gks)
    assert all(7000 < last[p][0] < 14000 for p in last if p not in gks)


def test_hidden_skill_not_in_events(match42):
    events_path, _, _, _ = match42
    assert "skill" not in events_path.read_text(encoding="utf-8").lower()


def test_generate_match_matches_written_file(match42):
    events_path, _, _, _ = match42
    events, _ = generate_match(42)
    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert [e.model_dump_json() for e in events] == lines


def test_generator_does_not_touch_the_model():
    source = Path(generator.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(joblib|xgboost|sklearn)\b", source, re.MULTILINE)
    assert ".joblib" not in source
