"""
Repetition metrics over the narrated lines of a match bundle, per language (no model calls).

    python scripts/line_variety.py data/bundle_seed3001.json [data/bundle_seed42.json ...] [--lines]

Each line is reduced to a skeleton: the player's name becomes <P>, a difficulty word (verifier.EASY_WORDS /
HARD_WORDS, with an intensifier before it) <T>, and every phrase-table wording (verifier.TABLE_PHRASES) <D>.
Per match and language:
  * near-dup: share of lines whose skeleton has a token similarity >= NEAR (difflib ratio) to another line's
    (the headline number: it catches lines that differ only in the name, tier, driver or outcome verb)
  * first6:   share of lines whose first 6 skeleton words match another line's
  * skel-dup: share of lines whose whole skeleton matches another line's exactly
  * mean-sim: mean pairwise similarity; top opener: the most common first 3 skeleton words
--lines prints every narrated line side by side per moment (EN, ES, EL) with its sentence frame
(narrator.frame_for, from the moment's slot among the commentated moments, as the workflow assigns it).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from xplain.contracts import MomentBundle
from xplain.narrator import frame_for
from xplain.verifier import EASY_WORDS, HARD_WORDS, LANGS, TABLE_PHRASES

NEAR = 0.8
INTENSIFIERS: dict[str, str] = {
    "en": r"\b(?:very|especially|extremely|particularly|really|quite|fairly)\s+",
    "es": r"\b(?:muy|especialmente|realmente|bastante|sumamente|extremadamente)\s+",
    "el": r"(?:πολύ|εξαιρετικά|ιδιαίτερα|αρκετά|πάρα)\s+",
}
_TIER = {lang: re.compile(rf"(?:{INTENSIFIERS[lang]})?(?:{'|'.join(EASY_WORDS[lang] + HARD_WORDS[lang])})",
                          re.IGNORECASE) for lang in LANGS}
_PHRASES = {lang: [re.compile(rf"(?<!\w){re.escape(p)}(?!\w)", re.IGNORECASE) for p in TABLE_PHRASES[lang]]
            for lang in LANGS}   # longest first
_WORD = re.compile(r"<[PTD]>|\w+")


def skeleton(line: str, name: str | None, lang: str) -> list[str]:
    """The line's words, lower case, with the name, the difficulty word and the driver wordings as slots."""
    s = line.replace(name, " <P> ") if name else line
    for pattern in _PHRASES[lang]:
        s = pattern.sub(" <D> ", s)
    s = _TIER[lang].sub(" <T> ", s)
    return [w if w.startswith("<") else w.lower() for w in _WORD.findall(s)]


def metrics(lines: list[tuple[str | None, str]], lang: str) -> dict:
    """lines: (player name, line) pairs of one match in one language."""
    sk = [skeleton(line, name, lang) for name, line in lines]
    n = len(sk)
    if n < 2:
        return {"n": n, "near_dup": 0.0, "first6": 0.0, "skel_dup": 0.0, "mean_sim": 0.0, "top_opener": ("", n)}
    sim = {(i, j): SequenceMatcher(None, sk[i], sk[j]).ratio() for i, j in itertools.combinations(range(n), 2)}

    def share(same) -> float:
        return sum(any(same(i, j) for j in range(n) if j != i) for i in range(n)) / n

    opener, count = Counter(" ".join(s[:3]) for s in sk).most_common(1)[0]
    return {
        "n": n,
        "near_dup": share(lambda i, j: sim[min(i, j), max(i, j)] >= NEAR),
        "first6": share(lambda i, j: sk[i][:6] == sk[j][:6]),
        "skel_dup": share(lambda i, j: sk[i] == sk[j]),
        "mean_sim": sum(sim.values()) / len(sim),
        "top_opener": (opener, count),
    }


def narrated_moments(bundle: dict) -> list[dict]:
    """The moments of a bundle (parsed JSON) that have a line, in seq order, each with its slot and frame."""
    live = sorted((m for m in bundle["moments"] if m["bundle"]["qualifies_live"]),
                  key=lambda m: m["bundle"]["moment"]["seq"])
    out = []
    for slot, m in enumerate(live):
        if m.get("line"):
            frame = frame_for(MomentBundle.model_validate(m["bundle"]), slot)
            out.append({"slot": slot, "frame": frame, "moment": m["bundle"]["moment"], "line": m["line"],
                        "narration": m["narration"], "tier": m["tier"]})
    return out


def report(path: Path, show_lines: bool) -> list[str]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    moments = narrated_moments(bundle)
    rows = []
    for lang in LANGS:
        r = metrics([(m["moment"]["player_name"], m["line"][f"line_{lang}"]) for m in moments], lang)
        opener, count = r["top_opener"]
        rows.append(f"{path.name} {lang.upper()} n={r['n']:2} near-dup {r['near_dup']:.0%} first6 {r['first6']:.0%} "
                    f"skel-dup {r['skel_dup']:.0%} mean-sim {r['mean_sim']:.2f} top opener {count}/{r['n']} "
                    f"'{opener}'")
    statuses = Counter(m["narration"]["status"] for m in bundle["moments"])
    first = sum(m["narration"]["status"] == "accepted" for m in moments)
    rows.append(f"{path.name} lines {len(moments)}: first try {first}, retried {statuses['retried']}, "
                f"rejected {statuses['rejected']}")
    if show_lines:
        frames = Counter(m["frame"] for m in moments)
        rows.append(f"{path.name} frames: " + ", ".join(f"{f} {n}" for f, n in frames.most_common()))
        for m in moments:
            mo = m["moment"]
            rows.append(f"[{mo['minute_label']}'] {m['frame']} | {mo['kind']} | {m['tier']['en']} | "
                        f"attempts {m['narration']['attempts']} | drivers_used {m['line']['drivers_used']}")
            rows += [f"  {lang.upper()}: {m['line'][f'line_{lang}']}" for lang in LANGS]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repetition metrics over a bundle's narrated lines.")
    parser.add_argument("bundles", type=Path, nargs="+")
    parser.add_argument("--lines", action="store_true", help="print every line side by side with its frame")
    args = parser.parse_args(argv)
    for path in args.bundles:
        print("\n".join(report(path, args.lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
