import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preview_recap as pr  # noqa: E402

from xplain.prompts import RECAP_EXAMPLES  # noqa: E402
from xplain.recap_editor import RecapResult  # noqa: E402
from xplain.trace import code_step, llm_step  # noqa: E402

FT = RECAP_EXAMPLES[0]


def test_card_lines_show_the_numbers_from_the_card():
    rows = pr.card_lines(FT.card)
    assert rows[0] == "full_time: Harbourne Athletic 2-1 Kellsworth Rovers (home_win)"
    assert "  goal 90+2' Harbourne Athletic: Tomas Avery (late)" in rows
    assert any(r.startswith("  Kellsworth Rovers:") and r.endswith("vs expected −15 (below)") for r in rows)


def test_result_lines_show_the_texts_rejections_and_trace():
    bad = FT.out.model_copy(update={"text_en": "x"})
    r = RecapResult(text=FT.out, attempts=2, reasons=["attempt 1: length: x"], latency_s=1.0,
                    trace=[llm_step("recap_editor", "gpt-4.1-mini", 1, True, 0.5),
                           code_step("recap_verifier", ok=False, reasons=["length: x"])], rejected=[bad])
    rows = pr.result_lines(r)
    assert rows[0] == "status retried, 2 attempt(s), 1.0 s"
    assert f"  EL: {FT.out.text_el}" in rows and "  rejected candidate 1:" in rows and "    EN: x" in rows
    assert "  recap_editor (llm) attempt 1: pass, gpt-4.1-mini, 0.50 s" in rows
    assert "  recap_verifier (code) attempt 1: fail | length: x" in rows


def test_usage_lines_count_reasoning_inside_output():
    from compare_models import CallRecord
    calls = [CallRecord(1.0, 3000, 500, 300), CallRecord(1.0, 1000, 100, None)]
    [row] = pr.usage_lines(calls, (2.00, 10.00))
    assert row == "tokens: 2 call(s), input 4000, output 600 (of which reasoning 300); cost 0.0140 USD"
    assert pr.usage_lines(calls, None)[0].endswith("cost n/a")
