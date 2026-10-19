"""The check registries (verifier.CHECKS, recap_verifier.RECAP_CHECKS) are the checks that run, and the trace records
them: a check listed in the Producer view is a check that ran on that line."""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from xplain import recap_verifier, verifier
from xplain.bundle import MatchBundle, Trace, build_bundle, check_labels, to_json
from xplain.config import PHRASE_MODES
from xplain.prompts import RECAP_EXAMPLES
from xplain.recap_verifier import RECAP_CHECK_LABELS, RECAP_CHECKS, verify_recap
from xplain.trace import TraceStep, code_step
from xplain.verifier import CHECK_LABELS, CHECKS, checks_for, verify

from .test_bundle import narrated
from .test_pipeline import WIDE, seed42_events, tiny_model, tiny_scorer  # noqa: F401  (fixtures)
from .test_recap_verifier import PLAYERS, check, sub
from .test_verifier import bare_out, line_out, make_bundle
from .test_workflow import GOOD

SRC = Path(__file__).resolve().parents[1] / "src" / "xplain"
PREFIX = re.compile(r'f?"([a-z_]+): ')


# --- the moment Verifier -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", PHRASE_MODES)
def test_verify_runs_exactly_the_registry_of_its_mode(mode):
    result = verify(line_out(), make_bundle(), mode)
    assert result.checks == [c.name for c in CHECKS if mode in c.modes] == [c.name for c in checks_for(mode)]


def test_the_mode_dependent_checks():
    assert [c.name for c in CHECKS if c.modes != frozenset(PHRASE_MODES)] == \
        ["phrases", "driver_expressed", "one_position_piece"]
    assert len(checks_for("hints")) == 18 and len(checks_for("strict")) == 17 and len(checks_for("free")) == 17


def test_every_reason_comes_from_a_check_that_ran():
    bad = [line_out(line_en="Weso Wendez tries a pass in the box at 94% in the 3rd minute; it fails. Twice."),
           bare_out(line_es="El pase fácil de Weso Wendez logra el éxito, porque sí, " + "muy " * 30 + "bien."),
           line_out(drivers_used=[1, 1, 7])]
    for out in bad:
        for mode in PHRASE_MODES:
            result = verify(out, make_bundle(), mode)
            assert result.reasons
            assert {r.split(":", 1)[0] for r in result.reasons} <= set(result.checks)


def test_the_source_emits_only_registry_names():
    """Every "name: ..." reason or note the module writes is a check of the registry, and every check writes one."""
    assert set(PREFIX.findall((SRC / "verifier.py").read_text(encoding="utf-8"))) == set(CHECK_LABELS)


# --- the Recap Verifier --------------------------------------------------------------------------

@pytest.mark.parametrize("example", RECAP_EXAMPLES, ids=lambda e: e.name)
def test_verify_recap_runs_every_registry_check(example):
    verdict = verify_recap(example.out, example.card, PLAYERS)
    assert verdict.ok and verdict.checks == [c.name for c in RECAP_CHECKS]


def test_recap_reasons_come_from_checks_that_ran_one_name_per_function():
    ft = RECAP_EXAMPLES[0]
    verdict = check(ft, **sub(ft, "en", ft.out.text_en.split()[0], "{star_name} deserved luck because 3 {oops}"))
    assert not verdict.ok
    names = {r.split(":", 1)[0] for r in verdict.reasons}
    assert {"no_numbers", "recap_slots", "banned", "recap_cause"} <= names <= set(verdict.checks)
    assert set(PREFIX.findall((SRC / "recap_verifier.py").read_text(encoding="utf-8"))) == set(RECAP_CHECK_LABELS)


# --- labels --------------------------------------------------------------------------------------

@pytest.mark.parametrize("labels", [CHECK_LABELS, RECAP_CHECK_LABELS], ids=["verifier", "recap_verifier"])
def test_labels_are_plain_words_and_unique(labels):
    for name, label in labels.items():
        assert label and label.lstrip("“")[0].isupper() and not label.endswith("."), name
        assert "_" not in label and not re.search(r"[\\\[\](){}|*+?^$]", label), (name, label)
    assert len(set(labels.values())) == len(labels)


def test_the_owners_wording():
    assert CHECK_LABELS["driver_direction"] == "Drivers agree with the rating"
    assert CHECK_LABELS["unsupported_fact"] == "Every fact comes from a driver"
    assert CHECK_LABELS["one_role_per_clause"] == "One reason or concession per clause"
    assert CHECK_LABELS["needs_driver"] == "A driver named whenever the rating needs one"
    assert CHECK_LABELS["length"] == f"At most {verifier.WORD_LIMIT} words"
    assert RECAP_CHECK_LABELS["recap_cause"] == "Says what happened, never why"
    assert RECAP_CHECK_LABELS["length"] == f"At most {recap_verifier.RECAP_WORD_LIMIT} words"


# --- the trace -----------------------------------------------------------------------------------

def test_only_verifier_steps_record_checks():
    assert code_step("verifier", checks=["sentence"]).checks == ["sentence"]
    assert code_step("recap_verifier", checks=["format"]).checks == ["format"]
    assert code_step("analyst").checks is None   # None: not recorded
    for agent in ("analyst", "explainer", "recap_filler"):
        with pytest.raises(ValidationError):
            code_step(agent, checks=["sentence"])
    with pytest.raises(ValidationError):
        TraceStep(agent="narrator", kind="llm", ok=True, model="m", latency_s=1.0, checks=[])


def test_a_built_bundle_records_the_checks_and_their_labels(seed42_events, tiny_scorer):  # noqa: F811
    from .test_workflow import RecapLLM, goalless_recap, run_full
    run_, _ = run_full(seed42_events, tiny_scorer, lambda i, f: GOOD(f), RecapLLM(goalless_recap),
                       scopes=("full_time", "half_time"))
    b = build_bundle(seed42_events, tiny_scorer, run_.outcomes, "hints", WIDE, recaps=run_.recaps)
    hints = [c.name for c in checks_for("hints")]
    for steps in b.trace.moments.values():
        verifiers = [s for s in steps if s.agent == "verifier"]
        assert [s.final for s in verifiers][-1] and all(s.checks == hints for s in verifiers)
        assert all(s.checks is None for s in steps if s.agent != "verifier")
    for steps in b.trace.recaps.values():
        assert [s.checks for s in steps if s.agent == "recap_verifier"] == [[c.name for c in RECAP_CHECKS]]
    assert b.trace.check_labels == {"verifier": {n: CHECK_LABELS[n] for n in hints}, "recap_verifier": RECAP_CHECK_LABELS}
    assert MatchBundle.model_validate_json(to_json(b)) == b


def test_a_check_without_a_label_is_rejected_and_old_traces_still_load(seed42_events, tiny_scorer):  # noqa: F811
    outcomes, _ = narrated(seed42_events, tiny_scorer, lambda i, f: GOOD(f))
    b = build_bundle(seed42_events, tiny_scorer, outcomes, "hints", WIDE)
    data = b.model_dump(mode="json")
    data["trace"]["check_labels"]["verifier"].pop("sentence")
    with pytest.raises(ValidationError):
        MatchBundle.model_validate(data)
    # a bundle written before checks were recorded: no checks, no labels
    data = b.model_dump(mode="json")
    data["trace"].pop("check_labels")
    for steps in data["trace"]["moments"].values():
        for s in steps:
            s.pop("checks")
    old = MatchBundle.model_validate(data)
    assert old.trace.check_labels == {} and all(s.checks is None for st in old.trace.moments.values() for s in st)


def test_a_reused_step_keeps_a_check_the_registry_no_longer_has():
    steps = {"m": [code_step("verifier", checks=["sentence", "retired_check"])]}
    labels = check_labels(steps, {})
    assert labels == {"verifier": {"sentence": CHECK_LABELS["sentence"], "retired_check": "retired_check"}}
    assert Trace(moments=steps, check_labels=labels).moments == steps
