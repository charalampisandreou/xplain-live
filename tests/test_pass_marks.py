import pytest
from pydantic import ValidationError

from xplain import analytics, pipeline
from xplain.clock import match_clock, second_half_start
from xplain.contracts import PassGeometry, PassMark, PlayerSummary
from xplain.pipeline import (
    analyse_match,
    marks_path,
    pass_marks,
    player_pass_map,
    player_summary,
    process_match,
)
from xplain.schema import PassEvent
from xplain.scoring import ModelLoadError, XPassScorer, find_default_model

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)

GEOMETRY = PassGeometry(start_x=30.0, start_y=40.0, end_x=45.0, end_y=35.0, length=15.8, ball_speed_ms=14.0)


def mark(seq: int, player_id: str | None, completed: bool, xp: float, *, t: float | None = None,
         team_id: str = "brk", outlier: str | None = None) -> PassMark:
    """A hand-built PassMark; difficulty, band and delta from the analytics functions."""
    t = 10.0 * seq if t is None else t
    d = analytics.difficulty(xp)
    c = match_clock(t, 1, None)
    return PassMark(
        event_id=f"m1:{seq}", seq=seq, t=t, minute=c.minute, minute_added=c.added, minute_label=c.label, period=1, team_id=team_id,
        player_id=player_id, player_name=None if player_id is None else f"Name {player_id}",
        pass_geometry=GEOMETRY, completed=completed, xp=xp, delta=analytics.delta(completed, xp),
        difficulty=d, band=analytics.difficulty_band(d), outlier=outlier,
        moment_id=None if outlier is None else f"moment:m1:{seq}",
    )


@pytest.fixture(scope="module")
def tiny_analysis(tiny_scorer, seed42_events):  # noqa: F811
    return analyse_match(seed42_events, tiny_scorer, WIDE)


# --- one mark per pass --------------------------------------------------------------------------

def test_one_mark_per_pass_in_seq_order(tiny_analysis, seed42_events):  # noqa: F811
    _, marks = tiny_analysis
    passes = [e for e in seed42_events if isinstance(e, PassEvent)]
    assert len(marks) == len(passes) > 0   # one mark per pass; the pass count depends on the generator
    assert [m.seq for m in marks] == sorted(p.seq for p in passes)
    assert len({m.event_id for m in marks}) == len(marks)


def test_delta_difficulty_band_from_analytics(tiny_analysis, seed42_events):  # noqa: F811
    _, marks = tiny_analysis
    start = second_half_start(seed42_events)
    for m in marks:
        assert m.delta == (1.0 if m.completed else 0.0) - m.xp
        assert m.difficulty == analytics.difficulty(m.xp)
        assert m.band == analytics.difficulty_band(m.difficulty, WIDE)
        c = match_clock(m.t, m.period, start)
        assert (m.minute, m.minute_added, m.minute_label) == (c.minute, c.added, c.label)


def test_outlier_links_match_moments(tiny_analysis):
    bundles, marks = tiny_analysis
    moments = {b.moment.moment_id: b.moment for b in bundles}
    linked = [m for m in marks if m.moment_id is not None]
    assert {m.moment_id for m in linked} == set(moments) and len(linked) == len(moments)
    for m in linked:
        moment = moments[m.moment_id]
        assert (m.event_id, m.outlier) == (moment.event_id, moment.kind)
        assert (m.xp, m.delta, m.difficulty, m.band) == (moment.xp, moment.delta, moment.difficulty, moment.band)
        assert m.pass_geometry == moment.pass_geometry
        assert (m.player_name, m.team_name) == (moment.player_name, moment.team_name)


def test_separate_calls_agree_with_shared_scoring(tiny_analysis, tiny_scorer, seed42_events):  # noqa: F811
    bundles, marks = tiny_analysis
    assert process_match(seed42_events, tiny_scorer, WIDE) == bundles
    assert pass_marks(seed42_events, tiny_scorer, WIDE) == marks


def test_deterministic_json(tiny_scorer, seed42_events):  # noqa: F811
    runs = [[m.model_dump_json() for m in pass_marks(seed42_events, tiny_scorer, WIDE)] for _ in range(2)]
    assert runs[0] == runs[1]


# --- per-player pass maps and summaries ---------------------------------------------------------

def test_player_pass_map_on_a_match(tiny_analysis):
    _, marks = tiny_analysis
    players = {m.player_id for m in marks if m.player_id is not None}
    total = 0
    for pid in players:
        own = player_pass_map(marks, pid)
        assert own and all(m.player_id == pid for m in own)
        assert [m.seq for m in own] == sorted(m.seq for m in own)
        early = player_pass_map(marks, pid, as_of_t=1800.0)
        assert all(m.t <= 1800.0 for m in early) and early == [m for m in own if m.t <= 1800.0]
        total += len(own)
    assert total + sum(m.player_id is None for m in marks) == len(marks)


def test_player_pass_map_never_returns_unattributed_passes():
    marks = [mark(1, "a", True, 0.8), mark(2, None, True, 0.9), mark(3, "a", False, 0.7), mark(4, "b", True, 0.6)]
    assert [m.seq for m in player_pass_map(marks, "a")] == [1, 3]
    assert player_pass_map(marks, None) == []
    assert player_pass_map(marks, "a", as_of_t=20.0) == [marks[0]]
    assert sum(len(player_pass_map(marks, p)) for p in ("a", "b")) + 1 == len(marks)


def test_player_summary_by_hand():
    marks = [mark(1, "a", True, 0.8), mark(2, "a", False, 0.9, outlier="surprise_failure"),
             mark(3, "a", True, 0.2, outlier="surprise_success"), mark(4, "b", True, 0.5),
             mark(5, "a", True, 0.7, t=100.0)]
    s = player_summary(marks, "a")
    # deltas: +0.2, -0.9, +0.8, +0.3  -> total 0.4
    assert (s.player_id, s.player_name, s.team_id) == ("a", "Name a", "brk")
    assert (s.passes, s.completed, s.outliers) == (4, 3, 2)
    assert s.completion_rate == pytest.approx(0.75)
    assert s.mean_xp == pytest.approx((0.8 + 0.9 + 0.2 + 0.7) / 4)
    assert s.pva_total == pytest.approx(0.4)
    assert s.pva_per_pass == pytest.approx(0.1)
    early = player_summary(marks, "a", as_of_t=20.0)
    assert (early.passes, early.completed, early.pva_total) == (2, 1, pytest.approx(0.2 - 0.9))
    assert player_summary(marks, "nobody") is None
    assert player_summary(marks, "a", as_of_t=5.0) is None
    assert isinstance(s, PlayerSummary) and PlayerSummary.model_validate_json(s.model_dump_json()) == s


# --- contract -----------------------------------------------------------------------------------

def test_pass_mark_validation():
    good = mark(1, "a", True, 0.3, outlier="surprise_success")
    data = good.model_dump()
    with pytest.raises(ValidationError, match="both be set"):
        PassMark.model_validate({**data, "moment_id": None})
    with pytest.raises(ValidationError, match="both be set"):
        PassMark.model_validate({**data, "outlier": None})
    PassMark.model_validate({**data, "outlier": None, "moment_id": None})
    for bad in ({"xp": 1.2}, {"xp": -0.1}, {"delta": 1.5}, {"difficulty": 101}, {"minute": 0}, {"period": 3},
                {"band": "impossible"}, {"extra": 1}):
        with pytest.raises(ValidationError):
            PassMark.model_validate({**data, **bad})
    with pytest.raises(ValidationError):
        good.xp = 0.5
    assert PassMark.model_validate_json(good.model_dump_json()) == good


# --- CLI ----------------------------------------------------------------------------------------

def test_cli_marks_and_player(tiny_scorer, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(pipeline, "find_default_model", lambda: ("unused", "unused"))
    monkeypatch.setattr(pipeline, "XPassScorer", lambda *_: tiny_scorer)
    match = tmp_path / "match.jsonl"
    pipeline.ensure_match_file(match, 42)
    capsys.readouterr()
    pipeline.main(["--file", str(match)])
    plain = capsys.readouterr().out

    marks_file, out = tmp_path / "marks.jsonl", tmp_path / "moments.jsonl"
    pipeline.main(["--file", str(match), "--marks-out", str(marks_file), "--out", str(out)])
    assert capsys.readouterr().out == plain  # file flags do not change what is printed
    lines = marks_file.read_text(encoding="utf-8").splitlines()
    n_passes = sum(isinstance(e, PassEvent) for e in pipeline.load_events(match))
    assert len(lines) == n_passes and marks_path(out).read_text(encoding="utf-8").splitlines() == lines
    first = PassMark.model_validate_json(lines[0])

    pipeline.main(["--file", str(match), "--player", first.player_id, "--as-of-minute", "30"])
    printed = capsys.readouterr().out
    assert printed.startswith(plain)
    assert f"player {first.player_id} (" in printed and "start -> end" in printed
    pipeline.main(["--file", str(match), "--player", "nobody"])
    assert "player nobody: no passes" in capsys.readouterr().out


# --- real model (skipped when the joblib file is missing) ---------------------------------------

def test_real_model_seed42_marks(seed42_events):  # noqa: F811
    try:
        model_path, feature_list_path = find_default_model()
    except ModelLoadError as exc:
        pytest.skip(f"xPass model not available ({exc})")
    if not model_path.is_file() or not feature_list_path.is_file():
        pytest.skip(f"xPass model not available: {model_path}")
    bundles, marks = analyse_match(seed42_events, XPassScorer(model_path, feature_list_path))
    # counts depend on the generator and the trained model, so none are pinned
    assert len(marks) == sum(isinstance(e, PassEvent) for e in seed42_events) and len(bundles) > 0
    assert sum(m.outlier is not None for m in marks) == len(bundles)
    assert {m.moment_id for m in marks if m.moment_id} == {b.moment.moment_id for b in bundles}
