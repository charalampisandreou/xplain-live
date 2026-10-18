"""
Fill in recap card fields added after a bundle was built, from the bundle's own passes, with no model call
(bundle.refresh_recap_cards). Every field the card already had is checked unchanged; each recap text is filled
again from the model's raw text (with slots) with the card's two-decimal values and checked again by the recap
verifier. Writes the bundle in place and prints its SHA-256 (swap the release asset when it is the demo bundle).

    python scripts/refresh_recap_cards.py data/bundle_seed3001.json
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from xplain.bundle import MatchBundle, refresh_recap_cards, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = MatchBundle.model_validate_json(args.bundle.read_text(encoding="utf-8"))
    args.bundle.write_text(to_json(refresh_recap_cards(bundle)), encoding="utf-8")
    print(f"{args.bundle}: {len(bundle.recaps)} recap card(s) refreshed; "
          f"SHA-256 {hashlib.sha256(args.bundle.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
