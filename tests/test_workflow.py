import asyncio
import json
import re

import pytest

from xplain import analytics
from xplain.narration import NarratorOut
from xplain.narrator import NarrationResult, Narrator, build_facts
from xplain.pipeline import gate_match
from xplain.workflow import MomentOutcome, main, outcome_lines, run_match, summary

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)

# A line that passes the Verifier for any moment: no number, no difficulty word (a note only). It cites the
# first driver pointing the tier's way, when there is one (needs_driver), and concedes it in every language
# with its first suggested wording (driver_expressed, role_parity; facts in hints mode).
NEUTRAL = NarratorOut(line_en="The ball moves on.", line_es="El balón sigue su camino.",
                      line_el="Η μπάλα συνεχίζει.", drivers_used=[])


def GOOD(facts: dict) -> NarratorOut:
    wanted = "raises" if facts["kind"] == "surprise_failure" else "lowers"
    used = [d for d in facts["drivers"] if d["direction"] == wanted][:1]
    if not used:
        return NEUTRAL
    d = used[0]
    hint = d["pieces"]["depth"]["suggested"] if "pieces" in d else d["suggested"]   # a position: its depth piece
    return NarratorOut(line_en=f"The ball moves on, even though it was {hint['en'][0]}.",
                       line_es=f"El balón sigue su camino, aunque fue {hint['es'][0]}.",
                       line_el=f"Η μπάλα συνεχίζει, παρόλο που ήταν {hint['el'][0]}.",
                       drivers_used=[used[0]["rank"]])


def BAD(facts: dict) -> NarratorOut:   # GOOD with a number: always rejected, for the number only
    good = GOOD(facts)
    return good.model_copy(update={"line_en": good.line_en.replace("The ball", "The 95% ball")})


class FakeAgent:
    """No network. answer(call_number, facts) gives a NarratorOut, or an exception to raise."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[dict] = []

    async def run(self, messages: str, *, options: dict):
        assert options["response_format"] is NarratorOut
        self.calls.append(json.loads(messages))
        out = self.answer(len(self.calls), self.calls[-1])
        if isinstance(out, Exception):
            raise out
        return type("Response", (), {"value": out})()


def run(events, scorer, answer, max_moments=None, narrator=None):
    agent = FakeAgent(answer)
    narrator = narrator or Narrator(agent, mode="hints")
    return asyncio.run(run_match(events, scorer, narrator, max_moments, cfg=WIDE)), agent


@pytest.fixture(scope="module")
def live(seed42_events, tiny_scorer):  # noqa: F811
    """The commentated moments of seed 42 on the tiny model, as the Analyst gates them."""
    moments = [m for _, m, why in gate_match(seed42_events, tiny_scorer, WIDE) if why]
    assert len(moments) >= 3
    return moments


def test_workflow_runs_one_outcome_per_commentated_moment(seed42_events, tiny_scorer, live):  # noqa: F811
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    assert [o.moment.moment_id for o in outcomes] == [m.moment_id for m in live]
    assert all(isinstance(o, MomentOutcome) and o.bundle.qualifies_live for o in outcomes)
    assert len(agent.calls) == len(live)
    assert all("minute" not in f and "xp" not in f for f in agent.calls)   # the model never sees numbers


def test_max_moments_limits_the_run(seed42_events, tiny_scorer, live):  # noqa: F811
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f), max_moments=2)
    assert [o.moment.moment_id for o in outcomes] == [m.moment_id for m in live[:2]]
    assert len(agent.calls) == 2
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f), max_moments=0)
    assert outcomes == [] and agent.calls == []


def test_accepted_line_passes_through(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f), max_moments=2)
    for o in outcomes:
        assert o.line == GOOD(build_facts(o.bundle, "hints")) and o.status == "accepted" and o.attempts == 1
        assert o.verdict.ok and o.reasons == []


def test_rejected_line_yields_no_line(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: BAD(f), max_moments=2)
    assert len(agent.calls) == 6   # each moment: the first call and two retries
    for o in outcomes:
        assert o.line is None and o.status == "rejected" and o.attempts == 3
        assert o.reasons == [f"attempt {i}: no_numbers: line_en has '95%'" for i in (1, 2, 3)]


def test_bad_then_good_is_retried(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: BAD(f) if i == 1 else GOOD(f), max_moments=2)
    assert [o.status for o in outcomes] == ["retried", "accepted"]
    assert agent.calls[1]["rejected_because"] == ["no_numbers: line_en has '95%'"]


def test_exception_in_one_moment_does_not_stop_the_next(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer,
                      lambda i, f: RuntimeError("connection reset") if i == 1 else GOOD(f), max_moments=3)
    assert [o.status for o in outcomes] == ["rejected", "accepted", "accepted"]
    assert outcomes[0].reasons == ["attempt 1: error RuntimeError: connection reset"]


def test_explainer_error_in_one_moment_does_not_stop_the_next(seed42_events, tiny_scorer, live,  # noqa: F811
                                                              monkeypatch):
    from xplain.explainer import ShapExplainer
    real = ShapExplainer.explain

    def explain(self, event, moment):
        if moment.moment_id == live[0].moment_id:
            raise ValueError("broken SHAP")
        return real(self, event, moment)

    monkeypatch.setattr(ShapExplainer, "explain", explain)
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f), max_moments=3)
    assert [o.moment.moment_id for o in outcomes] == [m.moment_id for m in live[:3]]   # still in seq order
    assert outcomes[0].bundle is None and outcomes[0].line is None
    assert outcomes[0].reasons == ["error in explainer: ValueError: broken SHAP"]
    assert [o.status for o in outcomes[1:]] == ["accepted", "accepted"]
    assert len(agent.calls) == 2


class RaisingNarrator(Narrator):
    """narrate() raises for the first moment (the real one never does): the executor's last guard."""

    raised = False

    async def narrate(self, bundle, slot=None):
        if not self.raised:
            self.raised = True
            raise RuntimeError("narrator down")
        return await super().narrate(bundle, slot)


def test_narrator_error_does_not_stop_the_next(seed42_events, tiny_scorer):  # noqa: F811
    agent = FakeAgent(lambda i, f: GOOD(f))
    outcomes, _ = run(seed42_events, tiny_scorer, None, max_moments=2,
                      narrator=RaisingNarrator(agent, mode="hints"))
    assert outcomes[0].status == "rejected" and outcomes[0].bundle is not None
    assert outcomes[0].reasons == ["error in narrator: RuntimeError: narrator down"]
    assert outcomes[1].status == "accepted"


class UncheckedNarrator(Narrator):
    """Returns BAD as if it had been accepted, so only the Verifier step can catch it."""

    async def narrate(self, bundle, slot=None):
        return NarrationResult(line=BAD(build_facts(bundle, "hints")), attempts=1, reasons=[], latency_s=0.0)


def test_verifier_step_drops_a_line_that_fails(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, None, max_moments=1,
                      narrator=UncheckedNarrator(FakeAgent(None), mode="hints"))
    (o,) = outcomes
    assert o.line is None and o.status == "rejected"
    assert not o.verdict.ok
    assert o.reasons == ["final check: no_numbers: line_en has '95%'"]


def test_outcome_lines_and_summary(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, lambda i, f: BAD(f) if i <= 4 else GOOD(f), max_moments=3)
    assert [o.status for o in outcomes] == ["rejected", "retried", "accepted"]
    m = outcomes[0].moment
    rejected, retried = outcome_lines(outcomes[0]), outcome_lines(outcomes[1])
    assert rejected[0].startswith(f"{m.minute_label:>5}' | ")                    # the minute from clock.match_clock
    assert f"xP {analytics.percent(m.xp)}%" in rejected[0]                       # the percent from code
    assert "  (no line)" in rejected
    good = outcomes[1].line
    assert good == GOOD(build_facts(outcomes[1].bundle, "hints"))
    assert retried[1:4] == [f"  EN: {good.line_en}", f"  ES: {good.line_es}", f"  EL: {good.line_el}"]
    # narrating time is measured wall clock: check its format, not its value
    assert re.fullmatch(r"3 moments: accepted 1, retried 1 \(accepted on a retry\), rejected 1; "
                        r"mean 1\.00 s per moment \(\d+\.\d\d s narrating\)", summary(outcomes, 3.0))
    assert summary([], 0.0).startswith("0 moments: accepted 0")


def test_cli_rejects_negative_max_moments():
    with pytest.raises(SystemExit):
        main(["--file", "unused.jsonl", "--max-moments", "-1"])


# --- trace --------------------------------------------------------------------------------------

def test_every_outcome_carries_its_trace(seed42_events, tiny_scorer, live):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, lambda i, f: BAD(f) if i == 1 else GOOD(f))
    first = outcomes[0]
    assert [(s.agent, s.kind, s.attempt, s.ok, s.final) for s in first.trace] == [
        ("analyst", "code", 1, True, False), ("explainer", "code", 1, True, False),
        ("narrator", "llm", 1, True, False), ("verifier", "code", 1, False, False),
        ("narrator", "llm", 2, True, False), ("verifier", "code", 2, True, False),
        ("verifier", "code", 2, True, True)]
    assert first.trace[0].notes == list(first.bundle.qualified_by)
    assert first.trace[3].reasons and first.trace[3].reasons[0].startswith("no_numbers")
    for o in outcomes[1:]:
        assert [s.agent for s in o.trace] == ["analyst", "explainer", "narrator", "verifier", "verifier"]
        assert o.trace[-1].final and o.trace[-1].ok


def test_a_rejected_moment_has_no_final_check_in_its_trace(seed42_events, tiny_scorer, live):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, lambda i, f: BAD(f), max_moments=1)
    trace = outcomes[0].trace
    assert outcomes[0].line is None and not any(s.final for s in trace)
    assert [s.attempt for s in trace if s.agent == "verifier"] == [1, 2, 3]
    assert not any(s.ok for s in trace if s.agent == "verifier")


def test_an_explainer_failure_is_in_the_trace(seed42_events, tiny_scorer, monkeypatch):  # noqa: F811
    from xplain.explainer import ShapExplainer

    def boom(self, event, moment):
        raise RuntimeError("no shap")

    monkeypatch.setattr(ShapExplainer, "explain", boom)
    outcomes, agent = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f), max_moments=1)
    assert agent.calls == []
    assert [(s.agent, s.ok) for s in outcomes[0].trace] == [("analyst", True), ("explainer", False)]
    assert "no shap" in outcomes[0].trace[-1].reasons[0]


# --- RecapEditor branch ---------------------------------------------------------------------------

def run_full(events, scorer, answer, recap_llm, scopes=("full_time",), max_moments=None):
    from xplain.recap_editor import RecapEditor
    from xplain.workflow import run_match_full

    agent = FakeAgent(answer)
    narrator = Narrator(agent, mode="hints")
    editor = RecapEditor(recap_llm, model="gpt-4.1-mini")
    return asyncio.run(run_match_full(events, scorer, narrator, max_moments, cfg=WIDE, recap_editor=editor,
                                      recap_scopes=scopes)), agent


class RecapLLM:
    """A fake RecapEditor model: answer(facts) gives a RecapOut or an exception."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    async def run(self, messages: str, *, options: dict):
        self.calls.append(json.loads(messages))
        out = self.answer(self.calls[-1])
        if isinstance(out, Exception):
            raise out
        return type("Response", (), {"value": out})()


def moments_of(outcomes):
    """What a run gives each moment, without wall times (they differ from run to run)."""
    return [(o.moment.moment_id, o.line, o.status, o.reasons,
             [st.model_copy(update={"latency_s": None}) for st in o.trace]) for o in outcomes]


def goalless_recap(facts):
    from xplain.recap import RecapOut
    h, a = facts["home"], facts["away"]
    phrase = {"above": ("completed more passes than expected", "completó más pases de lo esperado",
                        "ολοκλήρωσε περισσότερες πάσες από τις αναμενόμενες"),
              "below": ("completed fewer passes than expected", "completó menos pases de lo esperado",
                        "ολοκλήρωσε λιγότερες πάσες από τις αναμενόμενες"),
              "close": ("passed about as expected", "estuvo en línea con lo esperado", "κινήθηκε κοντά στα αναμενόμενα")}
    ph, pa = phrase[facts["passing"][h]], phrase[facts["passing"][a]]
    level = {"full_time": ("drew without a goal", "empataron sin goles", "έμειναν στην ισοπαλία χωρίς γκολ"),
             "half_time": ("are level and goalless at the break", "llegan al descanso igualados y sin goles",
                           "πάνε στα αποδυτήρια ισόπαλες και χωρίς γκολ")}[facts["scope"]]
    return RecapOut(text_en=f"{h} and {a} {level[0]}. {h} {ph[0]}; {a} {pa[0]}.",
                    text_es=f"{h} y {a} {level[1]}. {h} {ph[1]}; {a} {pa[1]}.",
                    text_el=f"Η {h} και η {a} {level[2]}. Η {h} {ph[2]}· η {a} {pa[2]}.",
                    claims_used=["result", f"passing:{h}", f"passing:{a}"])


def test_the_recap_branch_runs_beside_the_moments(seed42_events, tiny_scorer, live):  # noqa: F811
    llm = RecapLLM(goalless_recap)
    both, agent = run_full(seed42_events, tiny_scorer, lambda i, f: GOOD(f), llm, scopes=("full_time", "half_time"))
    alone, _ = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    assert moments_of(both.outcomes) == moments_of(alone)
    assert [r.scope for r in both.recaps] == ["full_time", "half_time"]
    for r in both.recaps:
        assert r.card is not None and r.card.scope == r.scope and r.text is not None and r.status == "accepted", r.reasons
        assert [s.agent for s in r.trace] == ["recap_editor", "recap_verifier"]
    assert [c["scope"] for c in llm.calls] == ["full_time", "half_time"]


def test_a_failing_recap_never_changes_the_moments(seed42_events, tiny_scorer, live):  # noqa: F811
    alone, _ = run(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    for answer in (lambda f: RuntimeError("recap model down"), lambda f: None):
        run_, _ = run_full(seed42_events, tiny_scorer, lambda i, f: GOOD(f), RecapLLM(answer))
        assert moments_of(run_.outcomes) == moments_of(alone)
        [r] = run_.recaps
        assert r.text is None and r.status == "rejected" and r.reasons and r.card is not None


def test_a_recap_card_that_cannot_be_built_is_reported(seed42_events, tiny_scorer, live, monkeypatch):  # noqa: F811
    import xplain.workflow as wf

    def boom(*a, **k):
        raise RuntimeError("no card")

    monkeypatch.setattr(wf, "recap_card", boom)
    llm = RecapLLM(goalless_recap)
    run_, _ = run_full(seed42_events, tiny_scorer, lambda i, f: GOOD(f), llm, max_moments=2)
    assert len(run_.outcomes) == 2 and all(o.line is not None for o in run_.outcomes)
    [r] = run_.recaps
    assert r.card is None and "no card" in r.reasons[0] and llm.calls == []
    assert [(s.agent, s.kind, s.ok) for s in r.trace] == [("recap_editor", "code", False)]


def test_without_a_recap_editor_there_is_no_recap(seed42_events, tiny_scorer, live):  # noqa: F811
    from xplain.workflow import run_match_full
    narrator = Narrator(FakeAgent(lambda i, f: GOOD(f)), mode="hints")
    result = asyncio.run(run_match_full(seed42_events, tiny_scorer, narrator, 1, cfg=WIDE))
    assert len(result.outcomes) == 1 and result.recaps == []
