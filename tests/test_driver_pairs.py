import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import driver_pairs as dp  # noqa: E402

from .test_verifier import make_bundle, make_driver  # noqa: E402

GROUND = ("height", "pass height", "Ground Pass", "along the ground")
WIDE = ("position", "pitch position", "middle_third|out_wide", "from the middle third, out wide")
CALM = ("under_pressure", "pressure", 0.0, "with no pressure on the passer")
PRESSED = ("under_pressure", "pressure", 1.0, "under pressure")


def live(bundle):
    return bundle.model_copy(update={"qualifies_live": True})


def test_mentionable_drops_not_narrated_and_wrong_direction():
    easy = make_bundle(drivers=[make_driver(1, *GROUND), make_driver(2, *CALM), make_driver(3, *PRESSED, "lowers")])
    assert [d.label for d in dp.mentionable(easy)] == ["along the ground"]


def test_pair_counts_only_commentated_moments():
    two = make_bundle(drivers=[make_driver(1, *GROUND), make_driver(2, *WIDE)])
    pairs, feature = dp.pair_counts([live(two), live(two), two])   # the third is not commentated
    assert pairs == {frozenset({"along the ground", "from the middle third, out wide"}): 2}
    assert feature == {"along the ground": "height", "from the middle third, out wide": "position"}


def test_parse_seeds():
    assert dp.parse_seeds("1-3") == [1, 2, 3] and dp.parse_seeds("7") == [7]
