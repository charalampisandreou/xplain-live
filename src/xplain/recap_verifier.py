"""
Recap Verifier: checks a RecapOut against its RecapCard. Deterministic, no LLM, no I/O.

It is separate from the Narrator's Verifier (verifier.py), which it does not change: it only reads that module's
tables (number words, banned claims, causal markers, sentence breaks) and adds its own.

Checks (every failing check adds a reason, prefixed with its name):
  * format:      each text is non-empty, has no line break and at most RECAP_MAX_SENTENCES sentences
  * length:      each text has at most RECAP_WORD_LIMIT words (the same for EN/ES/EL)
  * no_numbers:  no digit, "%", percent or minute word (verifier.NUMBER_WORDS), or number word
                 (RECAP_NUMBER_WORDS); the card shows the score, the minutes and "vs expected"
  * banned:      verifier.BANNED (claims we have no data for, model internals, intent and credit verbs) and
                 RECAP_BANNED (judgement and luck, dominance and control, tactics, pressure, possession, the model)
  * recap_cause: no causal marker (verifier.CAUSAL_MARKERS): a recap reports what happened, never why
  * recap_claims: what the text says is backed by the card:
      - claims_used holds only ids of recap.recap_claims(card), without duplicates
      - every used claim is expressed in every language (result, comeback, late, scorer, passing words)
      - story words need a used claim of their kind: comeback words a comeback claim; late words a late_winner or
        late_equaliser claim or a used scorer claim of a late goal; "expected" words a passing claim
      - the result words fit the result: no win or loss words for a draw or at half time, no draw words for a
        win, no lead words when level at half time and no level words for a lead; a clause with a win (lead)
        word does not name only the loser (the team behind), and a clause with a loss word not only the winner.
        Spanish "empate" is a draw word, except inside an explicit equaliser phrase (EQUALISER_PHRASES,
        "el empate de <scorer>"), which neither contradicts a win nor expresses a draw
      - a passing phrase (above / below / close to expected) names, in its own clause, only teams of that band
      - no player is named unless they scored in the scope and their scorer claim is used

Clauses are split at punctuation (, ; : · — and sentence ends), so the prompt asks for each team's passing
claim in its own clause.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from .recap import RecapCard, RecapOut, recap_claims
from .verifier import BANNED, CAUSAL_MARKERS, LANGS, NUMBER_CHARS, NUMBER_WORDS, SENTENCE_BREAK

RECAP_WORD_LIMIT = 50
RECAP_WORD_TARGET = 35
RECAP_MAX_SENTENCES = 2

RECAP_NUMBER_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|hundred|dozen)\b",
           r"\bnil\b"),
    "es": (r"\b(dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|veinte|treinta|cien|ciento)\b",
           r"\ba cero\b"),
    "el": (r"\b(δύο|τρία|τρεις|τέσσερα|τέσσερις|πέντε|έξι|επτά|εφτά|οκτώ|οχτώ|εννέα|εννιά|δέκα|έντεκα|δώδεκα"
           r"|είκοσι|τριάντα|εκατό)\b", r"\bμηδέν\b"),
}

# Greek column: drafted for the owner to skim. Spanish column: not reviewed by a native speaker.
RECAP_BANNED: dict[str, tuple[str, ...]] = {
    "en": (r"\bdeserv\w*", r"\bmerit\w*", r"\bunluck\w*", r"\bluck\w*", r"\bfortunate\b",
           r"\bdominat\w*", r"\bcontrol\w*", r"\bshould have\b", r"\btactic\w*", r"\bformation\b",
           r"\bpressur\w*", r"\bpressing\b", r"\bpossession\b", r"\bxp\b", r"\bxpass\b", r"\bmodel\w*",
           r"\bprobabilit\w*", r"\bpva\b", r"\bpremier league\b"),
    "es": (r"\bmerec\w*", r"\bmereci\w*", r"\bsuerte\b", r"\bafortunad\w*", r"\bdesafortunad\w*",
           r"\bdomin\w*", r"\bcontrol\w*", r"\bdeber[ií]a\w*", r"\bdebi[oó]\b", r"\bt[aá]ctic\w*",
           r"\bpresi[oó]n\w*", r"\bposesi[oó]n\b", r"\bxp\b", r"\bmodelo\w*", r"\bprobabilidad\w*"),
    "el": (r"άξιζ\w*", r"αξίζ\w*", r"\bδίκαι\w*", r"τύχη\w*", r"\bτυχερ\w*", r"άτυχ\w*", r"ατυχ\w*",
           r"κυριαρχ\w*", r"κυριάρχ\w*", r"έλεγχ\w*", r"ελέγχ\w*", r"έλεγξ\w*", r"\bέπρεπε\b", r"τακτικ\w*",
           r"πίεσ\w*", r"πρέσινγκ", r"\bκατοχ\w*", r"\bxp\b", r"μοντέλ\w*", r"πιθανότητ\w*"),
}

WIN_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\b(beat|beats|win|wins|won|victory|victorious|defeated|defeats)\b",),
    "es": (r"\bgan(a|an|ó|aron|ar)\b", r"\bvenc(e|en|ió|ieron|er)\b", r"\bvictoria\b", r"\bse impu(so|sieron)\b",
           r"\bse impone\w*", r"\btriunf\w*"),
    "el": (r"\bνίκ(η|ης|ησε|ησαν|άει|ούν)\b", r"κέρδισ\w*", r"κερδίζ\w*", r"επικράτ\w*", r"επικρατ\w*"),
}
LOSS_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\b(lost|lose|loses|losing|beaten|defeat)\b",),
    "es": (r"\bperd(ió|ieron|er)\b", r"\bpierd(e|en)\b", r"\bderrota\b(?! a\b)", r"\bca(e|en|yó|yeron) ante\b"),
    "el": (r"\bήττα\w*", r"\bηττ\w*", r"\bέχασ(ε|αν)\b", r"\bχάν(ει|ουν)\b"),
}
DRAW_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\bdraw\b", r"\bdrawn\b", r"\bdrew\b(?! level)", r"\bstalemate\b", r"\bshared the points\b",
           r"\ball square\b"),
    # strict: any "empate" is a draw word, except inside an explicit equaliser phrase (EQUALISER_PHRASES)
    "es": (r"\b(termin[oó]|acab[oó]|finaliz[oó]) en empate\b", r"\bempate final\b", r"\bempat(an|aron|ados|adas)\b",
           r"\btablas\b", r"\breparto de puntos\b", r"\bigualad[oa]s\b", r"\bempate\b"),
    "el": (r"ισοπαλ\w*", r"ισόπαλ\w*", r"\bμοιράστηκαν\b"),
}
LEVEL_WORDS: dict[str, tuple[str, ...]] = {
    "en": DRAW_WORDS["en"] + (r"\blevel\b", r"\bgoalless\b", r"\bscoreless\b"),
    "es": DRAW_WORDS["es"] + (r"\bsin goles\b",),
    "el": DRAW_WORDS["el"] + (r"\bχωρίς γκολ\b", r"\bλευκ(ή|ό)\b"),
}
# Explicit equaliser phrases: an "empate" inside one is the equalising goal, not a draw. They are taken out of the
# text before the result words are checked (equaliser_free), so they neither contradict a win nor express a draw.
# "el empate de <scorer>" is built per card from the scorers' names (full name or surname).
EQUALISER_PHRASES: dict[str, tuple[str, ...]] = {
    "en": (), "el": (),
    "es": (r"\bgol del empate\b",
           r"\b(logr(ó|a|ara|aba)|consigui(ó|era|eron)|consigue|pus(o|iera)|pone) el empate\b"),
}
LEAD_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\blead(s|ing)?\b", r"\bled\b", r"\bahead\b", r"\bin front\b"),
    "es": (r"\bventaja\b", r"\bpor delante\b", r"\bdelante en el marcador\b"),
    "el": (r"προηγ\w*", r"\bμπροστά\b"),
}
COMEBACK_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\bcome-?back\b", r"\bfrom behind\b", r"\bturn(ed|s)? (it|the game|the match) around\b",
           r"\bfought back\b", r"\bfight-?back\b"),
    "es": (r"\bremont\w*", r"\bdio la vuelta\b", r"\bvuelta al marcador\b", r"\bdarle la vuelta\b"),
    "el": (r"ανατροπ\w*", r"ανέτρεψ\w*", r"\bγύρισε το (παιχνίδι|ματς|σκορ)\b", r"\bαπό πίσω\b"),
}
LATE_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\blate\b", r"\bstoppage time\b", r"\badded time\b", r"\binjury time\b", r"\bat the death\b",
           r"\bclosing (stages|moments)\b", r"\blast-gasp\b", r"\bdying (moments|seconds)\b"),
    "es": (r"\btard\w*", r"\btiempo (añadido|de descuento)\b", r"\bdescuento\b", r"\bagóni\w*",
           r"\búltim(os|as) (instantes|compases)\b", r"\bin extremis\b"),
    "el": (r"καθυστερήσ\w*", r"\bφινάλε\b", r"\bεκπνοή\w*", r"\bτελευταίες στιγμές\b", r"\bαργά\b"),
}
PASSING_WORDS: dict[str, tuple[str, ...]] = {
    "en": (r"\bexpect\w*",), "es": (r"\besperad\w*", r"\besperab\w*"), "el": (r"αναμεν\w*", r"αναμέν\w*"),
}
BAND_WORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "above": {
        "en": (r"\b(more|better)(?: \w+)? than expected\b", r"\babove expect\w*"),
        "es": (r"\b(m[aá]s|mejor)(?: \w+)? de lo(s)? esperad\w*", r"\bpor encima de lo(s)? esperad\w*"),
        "el": (r"(περισσότερ|καλύτερ)\w*(?: \w+)? από τ\w* αναμενόμεν\w*", r"πάνω από τ\w* αναμενόμεν\w*"),
    },
    "below": {
        "en": (r"\b(fewer|less|worse)(?: \w+)? than expected\b", r"\bbelow expect\w*", r"\bshort of expect\w*"),
        "es": (r"\b(menos|peor)(?: \w+)? de lo(s)? esperad\w*", r"\bpor debajo de lo(s)? esperad\w*"),
        "el": (r"(λιγότερ|χειρότερ)\w*(?: \w+)? από τ\w* αναμενόμεν\w*", r"κάτω από τ\w* αναμενόμεν\w*"),
    },
    "close": {
        "en": (r"\bas expected\b", r"\bin line with expect\w*", r"\bclose to expect\w*"),
        "es": (r"\ben l[ií]nea con lo(s)? esperad\w*", r"\bcerca de lo(s)? esperad\w*", r"\bcomo se esperaba\b"),
        "el": (r"κοντά στ\w* αναμενόμεν\w*", r"όπως αναμενόταν", r"όσ\w*(?: \w+)? αναμενόταν",
               r"στα αναμενόμενα"),
    },
}

CLAUSE_SPLIT = re.compile(r"[,;:··–—.!?]")
WORD = re.compile(r"\w+")


def _rx(table: dict[str, tuple[str, ...]]) -> dict[str, list[re.Pattern[str]]]:
    return {lang: [re.compile(p, re.IGNORECASE) for p in ps] for lang, ps in table.items()}


_NUM = _rx({lang: NUMBER_WORDS[lang] + RECAP_NUMBER_WORDS[lang] for lang in LANGS})
_BANNED = _rx({lang: BANNED[lang] + RECAP_BANNED[lang] for lang in LANGS})
_CAUSE = _rx(CAUSAL_MARKERS)
_WIN, _LOSS, _DRAW = _rx(WIN_WORDS), _rx(LOSS_WORDS), _rx(DRAW_WORDS)
_LEVEL, _LEAD = _rx(LEVEL_WORDS), _rx(LEAD_WORDS)
_EQUALISER = _rx(EQUALISER_PHRASES)
_COMEBACK, _LATE, _PASSING = _rx(COMEBACK_WORDS), _rx(LATE_WORDS), _rx(PASSING_WORDS)
_BAND = {band: _rx(table) for band, table in BAND_WORDS.items()}


class RecapVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    reasons: list[str]
    notes: list[str] = []


def _find(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for p in patterns:
        if m := p.search(text):
            return m.group(0)
    return None


def _named(name: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE) is not None


def _player_named(name: str, text: str) -> bool:
    """The full name, or the surname alone."""
    return _named(name, text) or (" " in name and _named(name.rsplit(" ", 1)[1], text))


def equaliser_free(text: str, lang: str, scorers: Sequence[str]) -> str:
    """The text without its explicit equaliser phrases (EQUALISER_PHRASES, and "el empate de <scorer>" in Spanish)."""
    patterns = list(_EQUALISER[lang])
    if lang == "es":
        names = {n for s in scorers for n in (s, s.rsplit(" ", 1)[-1])}
        patterns += [re.compile(rf"\bel empate de {re.escape(n)}(?!\w)", re.IGNORECASE) for n in names]
    for p in patterns:
        text = p.sub(" ", text)
    return text


def clauses(text: str) -> list[str]:
    return [c for c in CLAUSE_SPLIT.split(text) if c.strip()]


def _format(out: RecapOut) -> list[str]:
    reasons = []
    for lang in LANGS:
        field, text = f"text_{lang}", getattr(out, f"text_{lang}")
        if not text.strip():
            reasons.append(f"format: {field} is empty")
            continue
        if "\n" in text:
            reasons.append(f"format: {field} has a line break")
        sentences = len(SENTENCE_BREAK[lang].findall(text.strip())) + 1
        if sentences > RECAP_MAX_SENTENCES:
            reasons.append(f"format: {field} has {sentences} sentences (at most {RECAP_MAX_SENTENCES})")
        words = len(text.split())
        if words > RECAP_WORD_LIMIT:
            reasons.append(f"length: {field} has {words} words (at most {RECAP_WORD_LIMIT})")
    return reasons


def _wording(out: RecapOut) -> list[str]:
    reasons = []
    for lang in LANGS:
        field, text = f"text_{lang}", getattr(out, f"text_{lang}")
        if m := NUMBER_CHARS.search(text):
            reasons.append(f"no_numbers: {field} has {m.group(0)!r}")
        elif hit := _find(_NUM[lang], text):
            reasons.append(f"no_numbers: {field} has {hit!r}")
        if hit := _find(_BANNED[lang], text):
            reasons.append(f"banned: {field} has {hit!r}")
        if hit := _find(_CAUSE[lang], text):
            reasons.append(f"recap_cause: {field} has {hit!r}; a recap says what happened, never why")
    return reasons


def _claims(out: RecapOut, card: RecapCard, players: Sequence[str]) -> tuple[list[str], list[str]]:
    claims = recap_claims(card)
    reasons, notes = [], []
    unknown = [c for c in out.claims_used if c not in claims]
    if unknown:
        reasons.append(f"recap_claims: unknown claim ids {unknown}")
    if len(set(out.claims_used)) != len(out.claims_used):
        reasons.append("recap_claims: claims_used has duplicates")
    used = [c for c in dict.fromkeys(out.claims_used) if c in claims]
    kinds = {c.split(":", 1)[0] for c in used}
    if "result" not in kinds:
        notes.append("recap_claims: the result claim is not used")
    home, away = card.home.name, card.away.name
    band_of = {home: card.home.band, away: card.away.band}
    scorers = {g.player_name for g in card.goals if g.player_name}
    late_scorer_used = any(g.late and f"scorer:{g.player_name}" in used for g in card.goals)
    winner = {"home_win": home, "away_win": away, "home_lead": home, "away_lead": away}.get(card.result)
    loser = ({home, away} - {winner}).pop() if winner else None
    full_time = card.scope == "full_time"

    for lang in LANGS:
        field, text = f"text_{lang}", getattr(out, f"text_{lang}")
        if not text.strip():
            continue

        def say(msg: str) -> None:
            reasons.append(f"recap_claims: {field} {msg}")

        # story words need a claim of their kind
        if (hit := _find(_COMEBACK[lang], text)) and "comeback" not in kinds:
            say(f"has {hit!r} but no comeback claim is used")
        if (hit := _find(_LATE[lang], text)) and not ({"late_winner", "late_equaliser"} & kinds or late_scorer_used):
            say(f"has {hit!r} but no late claim is used")
        if (hit := _find(_PASSING[lang], text)) and "passing" not in kinds:
            say(f"has {hit!r} but no passing claim is used")

        # the result words fit the result (draw and level words outside explicit equaliser phrases)
        result_text = equaliser_free(text, lang, sorted(scorers))
        win, loss = _find(_WIN[lang], text), _find(_LOSS[lang], text)
        if card.result in ("draw", "level", "home_lead", "away_lead") and (win or loss):
            say(f"has {(win or loss)!r} but " + ("the match was drawn" if card.result == "draw"
                                                 else "nothing is won or lost at half time"))
        if card.result in ("home_win", "away_win") and (hit := _find(_DRAW[lang], result_text)):
            say(f"has {hit!r} but the match was won")
        if card.result == "level" and (hit := _find(_LEAD[lang], text)):
            say(f"has {hit!r} but the teams are level")
        if card.result in ("home_lead", "away_lead") and (hit := _find(_LEVEL[lang], result_text)):
            say(f"has {hit!r} but a team leads")
        if winner:
            ahead_words = _WIN[lang] if full_time else _LEAD[lang]
            for clause in clauses(text):
                named = {n for n in (home, away) if _named(n, clause)}
                if named == {loser} and _find(ahead_words, clause):
                    say(f"gives the win or lead to {loser} in {clause.strip()!r}")
                if named == {winner} and _find(_LOSS[lang], clause):
                    say(f"gives the loss to {winner} in {clause.strip()!r}")

        # passing phrases name only teams of their band, in their own clause
        expressed_bands: set[str] = set()
        for clause in clauses(text):
            for band, table in _BAND.items():
                if not _find(table[lang], clause):
                    continue
                named = [n for n in (home, away) if _named(n, clause)]
                if not named:
                    say(f"has a passing phrase with no team in its clause: {clause.strip()!r}")
                wrong = [n for n in named if band_of[n] != band]
                if wrong:
                    say(f"says {wrong} passed '{band}' expected, but the card says "
                        f"{', '.join(f'{n}: {band_of[n]}' for n in wrong)}")
                expressed_bands |= {n for n in named if band_of[n] == band}

        # players: only claimed scorers
        for p in players:
            if _player_named(p, text) and not (p in scorers and f"scorer:{p}" in used):
                say(f"names {p}, who has no used scorer claim")

        # every used claim is expressed
        for c in used:
            kind, _, arg = c.partition(":")
            if kind == "result":
                table = {"home_win": _WIN, "away_win": _WIN, "draw": _LEVEL, "home_lead": _LEAD, "away_lead": _LEAD,
                         "level": _LEVEL}[card.result]
                if not _find(table[lang], result_text) and not (winner and _find(_LOSS[lang], text)):
                    say("does not express the result claim")
            elif kind == "comeback" and not (_find(_COMEBACK[lang], text) and _named(arg, text)):
                say(f"does not express {c!r} (a comeback word and the team)")
            elif kind in ("late_winner", "late_equaliser") and not _find(_LATE[lang], text):
                say(f"does not express {c!r} (no late word)")
            elif kind == "scorer" and not _player_named(arg, text):
                say(f"does not name the scorer of {c!r}")
            elif kind == "passing" and arg not in expressed_bands:
                say(f"does not express {c!r} ({arg}: {band_of.get(arg)} expected, in a clause naming the team)")
    return reasons, notes


def verify_recap(out: RecapOut, card: RecapCard, players: Sequence[str] = ()) -> RecapVerdict:
    """Every check on a RecapOut; players: every player name of the match (recap.roster)."""
    reasons = _format(out) + _wording(out)
    claim_reasons, notes = _claims(out, card, players)
    reasons += claim_reasons
    return RecapVerdict(ok=not reasons, reasons=reasons, notes=notes)
