"""
Take every position driver's label in a bundle again from explainer.POSITION_LABELS (e.g. after "from the attacking
third" became "in the attacking third"), with no model call (bundle.refresh_position_labels). Only those labels and
their evidence copies change; every line is checked again by the final Verifier. Writes the bundle in place and
prints how many labels changed and its SHA-256 (swap the release asset when it is the demo bundle).

    python scripts/refresh_position_labels.py data/bundle_seed3001.json
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from xplain.bundle import MatchBundle, refresh_position_labels, to_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = MatchBundle.model_validate_json(args.bundle.read_text(encoding="utf-8"))
    new = refresh_position_labels(bundle)
    changed = sum(a.label != b.label for old, now in zip(bundle.moments, new.moments)
                  for a, b in zip(old.bundle.explanation.drivers, now.bundle.explanation.drivers))
    args.bundle.write_text(to_json(new), encoding="utf-8")
    print(f"{args.bundle}: {changed} position label(s) changed; "
          f"SHA-256 {hashlib.sha256(args.bundle.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
