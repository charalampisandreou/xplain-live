import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import line_variety as lv  # noqa: E402

from xplain.narrator import FRAMES, frame_for  # noqa: E402

from .test_verifier import make_bundle  # noqa: E402

TEMPLATED = [  # the seed3001 lines the owner quoted: one skeleton, names swapped
    ("Fenan Morrford", "Fenan Morrford's pass was tricky mainly because it was lofted high, yet it finds a teammate."),
    ("Nevek Iversani", "Nevek Iversani's pass was tricky mainly because it was lofted high, yet it reaches a teammate."),
    ("Wesar Pellwick", "Wesar Pellwick's pass was tricky mainly because it was lofted high, yet it is completed."),
]
VARIED = [
    ("Garus Dornic", "Garus Dornic's pass was easy because it was played along the ground, but it goes astray."),
    ("Nevek Iversani", "Even though it was lofted high, Nevek Iversani's pass reaches a teammate."),
    ("Wesar Pellwick", "Lofted high, which made it difficult, yet Wesar Pellwick's pass comes off."),
]


def test_skeleton_replaces_name_tier_and_driver():
    assert lv.skeleton(TEMPLATED[0][1], "Fenan Morrford", "en")[:9] == [
        "<P>", "s", "pass", "was", "<T>", "mainly", "because", "it", "was"]
    assert "<D>" in lv.skeleton(TEMPLATED[0][1], "Fenan Morrford", "en")
    assert lv.skeleton("Η πάσα του Ivo Varga ήταν πολύ δύσκολη επειδή ήταν ψηλή.", "Ivo Varga", "el") == [
        "η", "πάσα", "του", "<P>", "ήταν", "<T>", "επειδή", "ήταν", "<D>"]


def test_templated_lines_are_near_duplicates_and_varied_lines_are_not():
    templated = lv.metrics(TEMPLATED, "en")
    assert templated["near_dup"] == 1.0 and templated["first6"] == 1.0
    assert templated["top_opener"] == ("<P> s pass", 3)
    varied = lv.metrics(VARIED, "en")
    assert varied["near_dup"] == 0.0 and varied["first6"] == 0.0 and varied["skel_dup"] == 0.0
    assert varied["mean_sim"] < templated["mean_sim"]


def test_fewer_than_two_lines():
    assert lv.metrics([], "en")["near_dup"] == 0.0
    assert lv.metrics(VARIED[:1], "en")["n"] == 1


def _entry(seq: int, live: bool, line: dict | None, status: str) -> dict:
    b = make_bundle().model_copy(update={"qualifies_live": live, "qualified_by": ("threshold",) if live else ()})
    b = b.model_copy(update={"moment": b.moment.model_copy(update={"seq": seq, "moment_id": f"moment:m1-{seq:05d}"})})
    return {"bundle": b.model_dump(mode="json"), "tier": {"en": "easy"}, "line": line,
            "narration": {"status": status, "attempts": 1}}


def test_slots_count_commentated_moments_in_seq_order(tmp_path, capsys):
    line = {"line_en": VARIED[0][1], "line_es": "El pase se pierde.", "line_el": "Η πάσα χάνεται.", "drivers_used": []}
    data = {"moments": [_entry(30, True, line, "accepted"), _entry(10, True, None, "rejected"),
                        _entry(20, False, None, "not_commentated"), _entry(40, True, line, "retried")]}
    moments = lv.narrated_moments(data)
    assert [m["slot"] for m in moments] == [1, 2]           # seq 10 is slot 0 (no line), seq 20 is not commentated
    assert all(m["frame"] in FRAMES for m in moments)
    assert moments[0]["frame"] == frame_for(make_bundle(), 1)
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert lv.main([str(path), "--lines"]) == 0
    out = capsys.readouterr().out
    assert "bundle.json EN n= 2" in out and "first try 1, retried 1, rejected 1" in out
    assert f"] {moments[0]['frame']} | surprise_failure | easy | attempts 1" in out
