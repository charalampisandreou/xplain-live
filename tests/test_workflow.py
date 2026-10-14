import asyncio
import json

import pytest

from xplain import analytics
from xplain.narration import NarratorOut
from xplain.narrator import NarrationResult, Narrator, build_facts
from xplain.pipeline import gate_match
from xplain.workflow import MomentOutcome, main, outcome_lines, run_match, summary

from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)

# A line that passes the Verifier for any moment: no number, no difficulty word (a note only), no driver
# phrase; it cites the first driver pointing the tier's way, when there is one (needs_driver).
NEUTRAL = NarratorOut(line_en="The ball moves on.", line_es="El balón sigue su camino.",
                      line_el="Η μπάλα συνεχίζει.", drivers_used=[])


def GOOD(facts: dict) -> NarratorOut:
    wanted = "raises" if facts["kind"] == "surprise_failure" else "lowers"
    ranks = [d["rank"] for d in facts["drivers"] if d["direction"] == wanted]
    return NEUTRAL.model_copy(update={"drivers_used": ranks[:1]})


def BAD(facts: dict) -> NarratorOut:   # a number: always rejected
    return GOOD(facts).model_copy(update={"line_en": "A 95% pass moves on."})


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
    assert len(agent.calls) == 4   # each moment: the first call and one retry
    for o in outcomes:
        assert o.line is None and o.status == "rejected" and o.attempts == 2
        assert o.reasons == ["attempt 1: no_numbers: line_en has '95%'", "attempt 2: no_numbers: line_en has '95%'"]


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

    async def narrate(self, bundle):
        if not self.raised:
            self.raised = True
            raise RuntimeError("narrator down")
        return await super().narrate(bundle)


def test_narrator_error_does_not_stop_the_next(seed42_events, tiny_scorer):  # noqa: F811
    agent = FakeAgent(lambda i, f: GOOD(f))
    outcomes, _ = run(seed42_events, tiny_scorer, None, max_moments=2,
                      narrator=RaisingNarrator(agent, mode="hints"))
    assert outcomes[0].status == "rejected" and outcomes[0].bundle is not None
    assert outcomes[0].reasons == ["error in narrator: RuntimeError: narrator down"]
    assert outcomes[1].status == "accepted"


class UncheckedNarrator(Narrator):
    """Returns BAD as if it had been accepted, so only the Verifier step can catch it."""

    async def narrate(self, bundle):
        return NarrationResult(line=BAD(build_facts(bundle, "hints")), attempts=1, reasons=[], latency_s=0.0)


def test_verifier_step_drops_a_line_that_fails(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, None, max_moments=1,
                      narrator=UncheckedNarrator(FakeAgent(None), mode="hints"))
    (o,) = outcomes
    assert o.line is None and o.status == "rejected"
    assert not o.verdict.ok
    assert o.reasons == ["final check: no_numbers: line_en has '95%'"]


def test_outcome_lines_and_summary(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = run(seed42_events, tiny_scorer, lambda i, f: BAD(f) if i <= 3 else GOOD(f), max_moments=3)
    assert [o.status for o in outcomes] == ["rejected", "retried", "accepted"]
    m = outcomes[0].moment
    rejected, retried = outcome_lines(outcomes[0]), outcome_lines(outcomes[1])
    assert rejected[0].startswith(f"{m.minute_label:>5}' | ")                    # the minute from clock.match_clock
    assert f"xP {analytics.percent(m.xp)}%" in rejected[0]                       # the percent from code
    assert "  (no line)" in rejected
    assert retried[1:4] == [f"  EN: {NEUTRAL.line_en}", f"  ES: {NEUTRAL.line_es}", f"  EL: {NEUTRAL.line_el}"]
    assert summary(outcomes, 3.0) == ("3 moments: accepted 1, retried 1 (accepted on the retry), rejected 1; "
                                      "mean 1.00 s per moment (0.00 s narrating)")
    assert summary([], 0.0).startswith("0 moments: accepted 0")


def test_cli_rejects_negative_max_moments():
    with pytest.raises(SystemExit):
        main(["--file", "unused.jsonl", "--max-moments", "-1"])
