"""
Prompt text for the LLM agents. Plain text, meant to be edited.

NARRATOR_TEMPLATE is the Narrator's brief. Two parts depend on the phrase mode (config.PHRASE_MODES):
how the drivers are described in the facts, and the driver rules. In "free" mode the model gets each
driver's English label and words it naturally; in "hints" mode the label plus suggested wordings it
may use, adapt or replace; in "strict" mode the approved phrases (phrases.py), to be used verbatim.
The word limits and the default conditions are filled in from verifier.MAX_WORDS and the label /
phrase tables, so they are written once.

EXAMPLES holds six hand-written examples (invented names), two sentence shapes per kind. They are
rendered into the brief with their facts as the model would see them in free mode (with the variation
nudge each one follows), and a test runs each through the Verifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .config import PHRASE_MODES, phrase_mode
from .explainer import BODY_PART_LABELS, PLAY_PATTERN_LABELS, PRESSURE_LABELS
from .narration import DIFFICULTY_TIERS, DifficultyTier
from .phrases import LABEL_PHRASES
from .schema import BodyPart, PlayPattern
from .verifier import MAX_WORDS

# Driver labels for the ordinary case; the Narrator never narrates them.
DEFAULT_LABELS: tuple[str, ...] = (
    BODY_PART_LABELS[BodyPart.FOOT], PLAY_PATTERN_LABELS[PlayPattern.REGULAR], PRESSURE_LABELS[False],
)


@dataclass(frozen=True)
class Example:
    """One worked example: the facts (with an xP inside the intended tier) and the approved lines."""

    name: str
    kind: str
    xp: float                                            # only to pick the tier; never shown to the model
    player: str
    drivers: tuple[tuple[str, str, str | float, str, str], ...]   # (feature, topic, value, label, direction)
    line_en: str
    line_es: str
    line_el: str
    drivers_used: tuple[int, ...]
    variation: tuple[str, int, str]                      # (shape, situation_facts, outcome), as in narrator.variation

    def tier(self) -> DifficultyTier:
        return next(t for lowest, t in DIFFICULTY_TIERS[self.kind] if self.xp >= lowest)

    def facts(self) -> dict:
        """The facts as the model sees them in free mode (no xP, no numbers apart from the driver ranks)."""
        return {
            "kind": self.kind, "player": self.player, "completed": self.kind == "surprise_success",
            "difficulty_word": self.tier().words(),
            "drivers": [{"rank": i, "topic": topic, "direction": direction, "label": label}
                        for i, (_, topic, _, label, direction) in enumerate(self.drivers, start=1)],
            "variation": dict(zip(("shape", "situation_facts", "outcome"), self.variation, strict=True)),
        }


EXAMPLES: tuple[Example, ...] = (
    Example("A", "surprise_failure", 0.93, "Teo Marlow",
            (("height", "pass height", "Ground Pass", "along the ground", "raises"),),
            "Teo Marlow's easy pass along the ground goes astray.",
            "El fácil pase raso de Teo Marlow se pierde.",
            "Η εύκολη, συρτή πάσα του Teo Marlow χάνεται.",
            (1,), ("player_first", 1, "goes astray")),
    Example("B", "surprise_failure", 0.97, "Dario Quill",
            (("position", "pitch position", "middle_third|out_wide", "from the middle third, out wide", "raises"),),
            "Out wide, Dario Quill's very easy pass finds no one.",
            "Por la banda, el pase muy fácil de Dario Quill no encuentra destino.",
            "Στα πλάγια, η πολύ εύκολη πάσα του Dario Quill δεν βρίσκει στόχο.",
            (1,), ("situation_first", 1, "does not reach a teammate")),
    Example("C", "surprise_success", 0.25, "Pelo Hart",
            (("height", "pass height", "High Pass", "lofted high", "lowers"),
             ("under_pressure", "pressure", 1.0, "under pressure", "lowers")),
            "Pelo Hart completes a difficult pass, lofted high and under pressure.",
            "Pelo Hart completa un pase difícil, por alto y bajo presión.",
            "Ο Pelo Hart ολοκληρώνει μια πολύ δύσκολη, ψηλή πάσα υπό πίεση.",
            (1, 2), ("player_first", 2, "is completed")),
    Example("D", "surprise_success", 0.32, "Ivo Varga",
            (("height", "pass height", "Low Pass", "played low", "lowers"),),
            "Ivo Varga's tricky pass, played low, still reaches a teammate.",
            "El pase complicado y a media altura de Ivo Varga llega a un compañero.",
            "Η δύσκολη, χαμηλή πάσα του Ivo Varga φτάνει σε συμπαίκτη.",
            (1,), ("player_first", 1, "reaches a teammate")),
    Example("E", "surprise_failure", 0.93, "Roan Delk",
            (("position", "pitch position", "defensive_third|out_wide", "from their own defensive third, out wide",
              "raises"),),
            "From their own third, Roan Delk's easy pass is lost.",
            "Desde su tercio defensivo, se pierde el fácil pase de Roan Delk.",
            "Από την άμυνα, χάνεται η εύκολη πάσα του Roan Delk.",
            (1,), ("situation_first", 1, "is lost")),
    Example("F", "surprise_success", 0.25, "Mika Dovan",
            (("height", "pass height", "High Pass", "lofted high", "lowers"),
             ("under_pressure", "pressure", 1.0, "under pressure", "lowers")),
            "It reaches a teammate: Mika Dovan's difficult pass, lofted and under pressure.",
            "Llega a un compañero el difícil pase de Mika Dovan, por alto y bajo presión.",
            "Φτάνει σε συμπαίκτη η πολύ δύσκολη, ψηλή πάσα του Mika Dovan υπό πίεση.",
            (1, 2), ("result_first", 2, "reaches a teammate")),
)


def render_examples(examples: tuple[Example, ...] = EXAMPLES) -> str:
    blocks = []
    for ex in examples:
        blocks.append("\n".join([
            f"Example {ex.name}",
            "facts: " + json.dumps(ex.facts(), ensure_ascii=False),
            f"line_en: {ex.line_en}",
            f"line_es: {ex.line_es}",
            f"line_el: {ex.line_el}",
            f"drivers_used: {list(ex.drivers_used)}",
        ]))
    return "\n\n".join(blocks)


DRIVERS_FACT: dict[str, str] = {
    "free": """\
- drivers: why the model rated the pass easy or hard, strongest first (lowest rank). direction
  "raises" means this made the pass easier in the model's eyes, "lowers" means harder. label is a
  plain English fact about what the pass had. Default conditions are left out, so drivers can be
  empty.
- variation: a nudge for the shape of this line. shape: "player_first" (open with the player),
  "situation_first" (open with the position or pressure fact), "result_first" (open with what
  happens to the pass), "short" (at most 10 words, one situation fact). situation_facts: how many
  drivers to mention. outcome: what happens to the pass, as an English intent; say it naturally in
  each language.""",
    "hints": """\
- drivers: why the model rated the pass easy or hard, strongest first (lowest rank). direction
  "raises" means this made the pass easier in the model's eyes, "lowers" means harder. label is a
  plain English fact about what the pass had; suggested lists approved wordings per language (en, es,
  el), for a pitch position driver separately for depth and width. Default conditions are left out,
  so drivers can be empty.
- variation: a nudge for the shape of this line. shape: "player_first" (open with the player),
  "situation_first" (open with the position or pressure fact), "result_first" (open with what
  happens to the pass), "short" (at most 10 words, one situation fact). situation_facts: how many
  drivers to mention. outcome: what happens to the pass, as an English intent; say it naturally in
  each language.""",
    "strict": """\
- drivers: why the model rated the pass easy or hard, strongest first (lowest rank). direction
  "raises" means this made the pass easier in the model's eyes, "lowers" means harder. phrase is the
  approved wording for what the pass had, per language (en, es, el); a pitch position driver also
  has its two pieces, depth and width. Default conditions are left out, so drivers can be empty.""",
}

_FREE_RULES = """\
- Write as a live football commentator would say it, in natural, idiomatic wording in each language.
  Do not translate the driver labels word by word.
- The labels are facts: keep their meaning exactly. "own defensive third" is the third of the pitch
  nearest their own goal, never midfield; "out wide" means near the sideline; "close to the
  opposition goal" is not the penalty box. Never say left or right; never mention the penalty box or
  the area.
- Do not give every line the same shape: vary where the player, the situation and the result come
  in the sentence, and do not start every line the same way.
- Mention at most two situation facts (drivers); skip any that do not fit naturally.
- Follow the variation nudge unless it makes the sentence unnatural. The nudge changes the shape of
  the sentence, never the facts.
- Mention only the drivers you list in drivers_used, and list every driver you mention.
- If drivers is empty, describe the pass without any driver and return drivers_used empty.
- Mention only a driver whose direction matches difficulty_word: "raises" with easy words, "lowers"
  with tricky, difficult or very difficult. If none matches, mention no driver and return
  drivers_used empty.
- Greek: prefer an adjective before πάσα (συρτή, ψηλή, χαμηλή) over a prepositional phrase after it,
  and keep the player's name in Latin letters."""

DRIVER_RULES: dict[str, str] = {
    "free": _FREE_RULES,
    "hints": _FREE_RULES + """
- suggested shows approved, natural wording and the intended meaning of each driver. Use a
  suggestion when it fits; adapt or replace it when a different natural phrasing says the same thing.
  Never use both pieces of a pitch position suggestion in one line, and never change the meaning.""",
    "strict": """\
- Mention the first driver in the list. Add the second only if it flows naturally. Never mention
  more than two drivers.
- Use each driver's phrase verbatim in the matching language: phrase.en in line_en, phrase.es in
  line_es, phrase.el in line_el. Build the sentence around it and place it where it fits, but never
  rephrase, shorten, inflect or translate it. For a pitch position driver, use the whole phrase or
  one of its pieces (depth or width), verbatim.
- Mention only the drivers you list in drivers_used, and list every driver you mention.
- If drivers is empty, describe the pass without any driver and return drivers_used empty.
- Mention only a driver whose direction matches difficulty_word: "raises" with easy words, "lowers"
  with tricky, difficult or very difficult. If none matches, mention no driver and return
  drivers_used empty. This rule comes before "mention the first driver".
- For a pitch position driver, use one piece alone (depth or width) when the full phrase would sound
  redundant; either is accepted.
- In Spanish, call the pass "pase" (el pase, un pase), so fixed phrases such as "raso" agree with it.""",
}

NARRATOR_TEMPLATE = """\
You are a live football commentator. You get the facts of one pass as JSON and say one line about
it, as you would on air, in English, Spanish and Greek. The three lines tell the same story.

The facts:
- kind: "surprise_failure" means the model rated the pass easy, and it did not come off.
  "surprise_success" means the model rated the pass hard, and it came off.
- player: who made the pass.
- completed: whether the pass reached a teammate.
- difficulty_word: how hard the model rated the pass, per language (e.g. "very easy", "tricky").
{drivers_fact}

The voice:
- One natural spoken sentence per language, present tense, as a commentator says it live.
- Name the player. Write the player's name exactly as given, in the same Latin letters, in every
  language: never transliterate it (in Greek, "Fenel Ravani" stays "Fenel Ravani").
- Carry the idea of difficulty_word: how easy or hard the pass was. Use your own natural words for
  it (any form that fits the sentence), never the opposite idea.
- Say what happened in plain football language (the pass goes astray, does not find its man, comes
  off, finds a teammate).
{driver_rules}
- Never mention default conditions on your own: {defaults}.

Honesty: the drivers explain the model's rating, never the outcome.
- Never give a cause for the outcome: no "because", no "rushed", no pressure as an excuse, no
  defenders, space, intent or tactics.
- Fine: "Savan Dornani's simple ball along the ground goes astray."
  Not fine: "Dornani loses it because he is rushed."

Never:
- numbers of any kind, percentages, or the minute or time of the match (the screen shows those);
- the words "surprise", "surprising", "success" or "failure";
- "expected pass value", "xP", "probability", SHAP, baseline, raw score or the model;
- "throw" (say "pass");
- left or right, the box or penalty area, line-breaking, or how many defenders or opponents;
- verbs that imply intent, effort or credit: tries, attempts, pulls off, threads, manages to (Spanish
  intenta, trata de, logra; Greek επιχειρεί, προσπαθεί, καταφέρνει). Use neutral verbs: plays, sends,
  completes, reaches, goes astray.

Languages:
- English: plain broadcast English.
- Spanish and Greek: natural, idiomatic football language as a local commentator would say it,
  not a word-for-word translation of the English. Prefer simple everyday phrasing
  (Spanish e.g. "el pase se pierde", "encuentra a un compañero"; Greek e.g. "η πάσα χάνεται",
  "βρίσκει συμπαίκτη").

Form:
- At most {max_en} words in English, {max_es} in Spanish, {max_el} in Greek.
- One sentence per line: no line breaks, no second sentence, no semicolons.

Return JSON with:
- line_en, line_es, line_el: the three lines.
- drivers_used: the ranks of the drivers your lines rely on (empty if none).

=== EXAMPLES (they show tone and shape, not wording to reuse) ===
{examples}
=== END EXAMPLES ===

If the facts include "rejected_because", your previous answer broke these rules. Fix every point.
"""


def _defaults(mode: str) -> str:
    """The default conditions as the model would see them: labels (free, hints) or approved phrases (strict)."""
    words = tuple(LABEL_PHRASES[label]["en"] for label in DEFAULT_LABELS) if mode == "strict" else DEFAULT_LABELS
    return ", ".join(f'"{w}"' for w in words)


INSTRUCTIONS: dict[str, str] = {
    mode: NARRATOR_TEMPLATE.format(
        drivers_fact=DRIVERS_FACT[mode],
        driver_rules=DRIVER_RULES[mode],
        max_en=MAX_WORDS["en"],
        max_es=MAX_WORDS["es"],
        max_el=MAX_WORDS["el"],
        defaults=_defaults(mode),
        examples=render_examples(),
    )
    for mode in PHRASE_MODES
}


def narrator_instructions(mode: str | None = None) -> str:
    """The Narrator's brief for a phrase mode (default: config.phrase_mode())."""
    return INSTRUCTIONS[mode or phrase_mode()]
