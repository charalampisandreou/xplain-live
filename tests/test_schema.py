import pytest
from pydantic import TypeAdapter, ValidationError

from xplain.schema import Event, MatchMetaEvent, PlayerInfo, PlayerLoadEvent


def make_players(n: int = 22) -> list[PlayerInfo]:
    positions = ["GK"] + ["DEF"] * 4 + ["MID"] * 4 + ["FWD"] * 2
    return [
        PlayerInfo(player_id=f"t{k}-p{i + 1}", name=f"Player {k}{i}", position=positions[i % 11],
                   shirt_number=i + 1, team_id=f"t{k}")
        for k in (1, 2) for i in range(11)
    ][:n]


def make_meta(**kw) -> MatchMetaEvent:
    base = dict(event_id="m1-00000", match_id="m1", seq=0, t=0.0, period=1, team_id="t1",
                home_team_id="t1", away_team_id="t2", home_name="Home", away_name="Away",
                seed=1, generator_version="0.1.0", players=make_players())
    base.update(kw)
    return MatchMetaEvent(**base)


def make_load(**kw) -> PlayerLoadEvent:
    base = dict(event_id="m1-00001", match_id="m1", seq=1, t=300.0, period=1, team_id="t1",
                player_id="t1-p1", distance_m=1200.5, top_speed_ms=8.25, sprint_count=3)
    base.update(kw)
    return PlayerLoadEvent(**base)


def test_player_info_validation():
    with pytest.raises(ValidationError):
        PlayerInfo(player_id="a", name="A", position="ST", shirt_number=9, team_id="t1")
    with pytest.raises(ValidationError):
        PlayerInfo(player_id="a", name="A", position="FWD", shirt_number=100, team_id="t1")


def test_match_meta_requires_22_players():
    assert len(make_meta().players) == 22
    with pytest.raises(ValidationError):
        make_meta(players=make_players(21))


def test_player_load_validation():
    assert make_load().type == "player_load"
    with pytest.raises(ValidationError):
        make_load(player_id=None)
    with pytest.raises(ValidationError):
        make_load(top_speed_ms=12.5)
    with pytest.raises(ValidationError):
        make_load(distance_m=-1.0)
    with pytest.raises(ValidationError):
        make_load(sprint_count=-1)
    with pytest.raises(ValidationError):
        make_load(heart_rate=150)  # extra forbidden


def test_event_union_roundtrip():
    adapter = TypeAdapter(Event)
    for ev in (make_meta(), make_load()):
        again = adapter.validate_json(ev.model_dump_json())
        assert type(again) is type(ev) and again == ev
