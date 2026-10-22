"""
Line check: a line a producer edited or wrote in the Producer view ("Edit line" / "Write line"), checked by the code
Verifier before it may go on air. AI writes, a human edits, code checks, then air.

POST /api/line/check (api.py), body {"match": name, "moment_id": id, "line": {"en", "es", "el"}}. Always on: no model
call, no cost, nothing stored. The moment comes from the served bundle as stored (retake.find_moment: any moment, so
an uncommentated one too). Each line has its whitespace collapsed to single spaces (the plate shows one line); the
answer gives the lines as checked, and only those go on air.

The phrasing rules keep the model in bounds; a human writes as they like, so a human line runs only the fact checks,
HUMAN_CHECKS: the Verifier's own functions where they exist (verifier.check_difficulty, check_banned with the claim
families of verifier.BANNED_FACTS only, check_driver_direction, check_outcome_cause, check_unsupported_facts,
check_lead_driver), plus three the model never needs, here: filled (a line in every language), numbers (its lines
carry none; a human's must match the data) and names (it names only the passer). Length is a note (the plate may not
fit the line), never a reason. drivers_used is inferred, not declared: the moment's narrated drivers whose wording
(phrases.wordings) appears in any of the lines.

The answer: ok, line {en, es, el} (as checked), reasons, notes, checks (the names that ran), labels (their plain
words), drivers_used, steps (trace.TraceStep dicts: a producer step, kind "human", then the final Verifier step as
retake.final_step makes it).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from .contracts import MomentBundle
from .narration import NarratorOut
from .phrases import wordings
from .retake import final_step, find_moment
from .trace import TraceStep
from .verifier import (BANNED_FACTS, CHECK_LABELS, LANGS, VerifierResult, check_banned, check_difficulty,
                       check_driver_direction, check_lead_driver, check_length, check_outcome_cause,
                       check_unsupported_facts, has_phrase, narrated_drivers)

MAX_CHARS = 400   # per line, enforced by the request model in api.py

# A number in a line: "28", "27.5", "27,5", the added-time label "90+4". A sign before it is not part of it.
NUMBER = re.compile(r"\d+(?:\+\d+)?(?:[.,]\d+)?")
# What makes a number a percent, right after it.
PERCENT_AFTER: dict[str, re.Pattern[str]] = {
    "en": re.compile(r"\s*(?:%|per ?cent\b)", re.IGNORECASE),
    "es": re.compile(r"\s*(?:%|por ?ciento\b)", re.IGNORECASE),
    "el": re.compile(r"\s*(?:%|τοις εκατό)", re.IGNORECASE),
}
_BANNED_FACTS_RE = {lang: [re.compile(p, re.IGNORECASE) for p in patterns] for lang, patterns in BANNED_FACTS.items()}


@dataclass(frozen=True)
class Roster:
    """The names a line may hold: every player of the match and both clubs (the bundle's meta)."""
    players: tuple[str, ...]
    clubs: tuple[str, ...]


def roster(bundle: dict) -> Roster:
    meta = bundle.get("meta", {})
    players = tuple(p["name"] for p in meta.get("players", []) if p.get("name"))
    clubs = tuple(t["name"] for t in (meta.get("home"), meta.get("away")) if t and t.get("name"))
    return Roster(players=players, clubs=clubs)


def _lines(out: NarratorOut) -> dict[str, str]:
    return {lang: getattr(out, f"line_{lang}") for lang in LANGS}


def shown_percent(x: float, decimals: int = 0) -> float:
    """x as the page shows it in percent: app.js Math.round(100 * x), rounding half up; to `decimals` places."""
    scale = 10 ** decimals
    return math.floor(100 * x * scale + 0.5) / scale


def check_filled(out: NarratorOut) -> list[str]:
    return [f"filled: line_{lang} is empty" for lang, line in _lines(out).items() if not line.strip()]


def check_numbers(out: NarratorOut, bundle: MomentBundle) -> list[str]:
    """Every number matches the data: a percent is the pass's xP or its PVA (|delta|) as the page shows them (rounded
    half up, to the decimals the line gives); any other number is the moment's minute (its label, "90+4", or the
    base minute)."""
    m = bundle.moment
    xp, pva = shown_percent(m.xp), shown_percent(abs(m.delta))
    reasons = []
    for lang, line in _lines(out).items():
        for n in NUMBER.finditer(line):
            text = n.group(0)
            if PERCENT_AFTER[lang].match(line, n.end()):
                places = len(re.split(r"[.,]", text)[1]) if re.search(r"[.,]", text) else 0
                ok = "+" not in text and any(shown_percent(x, places) == float(text.replace(",", "."))
                                             for x in (m.xp, abs(m.delta)))
                if not ok:
                    reasons.append(f"numbers: line_{lang} has {text + '%'!r}, but the pass's xP is {xp:g}% and its "
                                   f"PVA {pva:g}%")
            elif text not in (m.minute_label, str(m.minute)):
                reasons.append(f"numbers: line_{lang} has {text!r}, but the moment's minute is {m.minute_label}")
    return reasons


def _name_parts(name: str) -> list[str]:
    """A name and its words of three letters or more, the full name first."""
    return [name, *(w for w in name.split() if len(w) >= 3 and w != name)]


def _spans(line: str, phrase: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", line, re.IGNORECASE)]


def check_names(out: NarratorOut, bundle: MomentBundle, names: Roster) -> list[str]:
    """A player or club named is the moment's passer or the passer's club: any other name of the roster, in full or
    by one of its words ("Kestley", "Marrowfield"), whole words, case-insensitive, fails. A word the passer or the
    club share with another name is allowed."""
    m = bundle.moment
    own = [n for n in (m.player_name, m.team_name) if n]
    allowed = {p.casefold() for n in own for p in _name_parts(n)}
    others = [n for n in (*names.players, *names.clubs) if n and n not in own]
    reasons = []
    for lang, line in _lines(out).items():
        seen: list[tuple[int, int]] = []
        for name in others:
            for part in _name_parts(name):
                if part.casefold() in allowed:
                    continue
                for a, b in _spans(line, part):
                    if not any(x <= a and b <= y for x, y in seen):
                        seen.append((a, b))
                        reasons.append(f"names: line_{lang} names {line[a:b]!r} ({name}), but the pass is "
                                       f"{m.player_name or 'by an unnamed player'}'s ({m.team_name or 'no club'})")
    return reasons


def length_notes(out: NarratorOut) -> list[str]:
    return [f"{r}; it may not fit the plate" for r in check_length(out)]


HumanRun = Callable[[NarratorOut, MomentBundle, Roster], tuple[list[str], list[str]]]


@dataclass(frozen=True)
class HumanCheck:
    """One check of a human line: its name (the prefix of its reasons), its plain-words label and the function."""
    name: str
    label: str
    run: HumanRun


def _fact(fn: Callable[[NarratorOut, MomentBundle], list[str]]) -> HumanRun:
    return lambda out, bundle, names: (fn(out, bundle), [])


# The order is the order of the reasons. Labels: English UI only (the Producer view).
HUMAN_CHECKS: tuple[HumanCheck, ...] = (
    HumanCheck("filled", "A line in every language", lambda out, bundle, names: (check_filled(out), [])),
    HumanCheck("numbers", "Numbers match the data", _fact(check_numbers)),
    HumanCheck("names", "Only this pass's player and club named",
               lambda out, bundle, names: (check_names(out, bundle, names), [])),
    HumanCheck("difficulty", CHECK_LABELS["difficulty"], lambda out, bundle, names: check_difficulty(out, bundle)),
    HumanCheck("banned", "No claims we have no data for",
               lambda out, bundle, names: (check_banned(out, _BANNED_FACTS_RE), [])),
    HumanCheck("driver_direction", CHECK_LABELS["driver_direction"], _fact(check_driver_direction)),
    HumanCheck("outcome_cause", CHECK_LABELS["outcome_cause"], _fact(check_outcome_cause)),
    HumanCheck("unsupported_fact", CHECK_LABELS["unsupported_fact"], _fact(check_unsupported_facts)),
    HumanCheck("lead_driver", CHECK_LABELS["lead_driver"], _fact(check_lead_driver)),
    HumanCheck("length", "Fits the plate (a warning, never blocks)",
               lambda out, bundle, names: ([], length_notes(out))),
)
HUMAN_CHECK_LABELS: dict[str, str] = {c.name: c.label for c in HUMAN_CHECKS}


def verify_human(out: NarratorOut, bundle: MomentBundle, names: Roster) -> VerifierResult:
    """Run every HUMAN_CHECKS entry and collect every failing reason; ok when there are none. out.drivers_used is the
    drivers the line names (inferred_drivers), not a model's claim."""
    reasons: list[str] = []
    notes: list[str] = []
    for check in HUMAN_CHECKS:
        r, n = check.run(out, bundle, names)
        reasons += r
        notes += n
    return VerifierResult(ok=not reasons, reasons=reasons, notes=notes, checks=[c.name for c in HUMAN_CHECKS])


def inferred_drivers(moment: MomentBundle, lines: dict[str, str]) -> list[int]:
    """The ranks of the moment's narrated drivers that any of the lines names by one of its wordings, in rank order."""
    return [rank for rank, d in sorted(narrated_drivers(moment).items())
            if any(has_phrase(lines[lang], w) for lang in LANGS for w in wordings(d)["all"][lang])]


def check_line(bundle: dict, moment_id: str, lines: dict[str, str]) -> dict:
    """Check a producer's line for one moment of a served bundle; raises retake.RetakeRefused (404) for an unknown
    moment."""
    moment, _ = find_moment(bundle, moment_id)
    lines = {lang: " ".join((lines.get(lang) or "").split()) for lang in LANGS}
    drivers = inferred_drivers(moment, lines)
    out = NarratorOut(line_en=lines["en"], line_es=lines["es"], line_el=lines["el"], drivers_used=drivers)
    verdict = verify_human(out, moment, roster(bundle))
    steps = [TraceStep(agent="producer", kind="human", ok=True), final_step(verdict, 1)]
    return {"moment_id": moment_id, "ok": verdict.ok, "line": lines, "reasons": list(verdict.reasons),
            "notes": list(verdict.notes), "checks": list(verdict.checks),
            "labels": {name: HUMAN_CHECK_LABELS[name] for name in verdict.checks}, "drivers_used": drivers,
            "steps": [s.model_dump(mode="json") for s in steps]}
