"""
Model comparison for the Narrator (measurement only: nothing here changes the shared Narrator, prompt or Verifier).

    python -u scripts/compare_models.py [--file data/match_seed3001.jsonl] [--models gpt-4.1-mini gpt-4.1 ...]
    python -u scripts/compare_models.py --report-only      # rebuild the report from the saved scratch bundles

For each deployment the same commentated moments are narrated exactly as the bundle builder does it
(bundle.narrate_match -> workflow.run_match -> Narrator, hints mode, narrator.MAX_ATTEMPTS, the current prompt,
frames, phrase table and Verifier), and a scratch bundle is written per model under reports/model_compare/.
The only per-model difference is the request parameters: ModelAgent wraps the Foundry agent and rewrites the
options the Narrator sends (MODEL_PARAMS), and times every call and records its token usage.

Report (reports/model_compare/report.md): first try / retried / rejected with the Verifier's reasons, the
repetition metrics of scripts/line_variety.py, latency p50/p95 per call and per line, tokens per line, a rough
cost per bundle, then every English line side by side per moment (ES and EL in an appendix).
Confound: gpt-6.1-sol runs at temperature 1 (it rejects any other value); the others at narrator.TEMPERATURE.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_variety as lv  # noqa: E402

from xplain.narrator import MAX_ATTEMPTS, NARRATOR_TIMEOUT_S, TEMPERATURE, Narrator, NarratorAgent  # noqa: E402
from xplain.verifier import LANGS  # noqa: E402

MODE = "hints"
OUT_DIR = Path("reports/model_compare")
DEFAULT_MATCH = Path("data/match_seed3001.jsonl")


@dataclass(frozen=True)
class ModelParams:
    """The request parameters of one deployment. None: the parameter is not sent."""
    temperature: float | None
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = None
    timeout_s: float = NARRATOR_TIMEOUT_S


MODEL_PARAMS: dict[str, ModelParams] = {
    "gpt-4.1-mini": ModelParams(temperature=TEMPERATURE),   # the baseline, as the Narrator runs today
    "gpt-4.1": ModelParams(temperature=TEMPERATURE),
    # rejects temperature != 1 (so none is sent: the default 1); a reasoning model, hence the larger budget
    "gpt-6.1-sol": ModelParams(temperature=None, max_completion_tokens=2000, reasoning_effort="low", timeout_s=60.0),
}
# USD per 1M tokens (input, output), Global Standard list prices; None: not known here, fill in to get a cost.
PRICES: dict[str, tuple[float, float] | None] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-6.1-sol": None,
}


def adapt_options(options: dict[str, Any], params: ModelParams) -> dict[str, Any]:
    """The Narrator's options rewritten for one deployment: its temperature (or none), and the extra parameters."""
    out = {k: v for k, v in options.items() if k != "temperature"}
    if params.temperature is not None:
        out["temperature"] = params.temperature
    if params.max_completion_tokens is not None:
        out["max_tokens"] = params.max_completion_tokens   # the chat client sends it as max_completion_tokens
    if params.reasoning_effort is not None:
        out["reasoning_effort"] = params.reasoning_effort
    return out


@dataclass
class CallRecord:
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    error: str | None = None


@dataclass
class ModelAgent:
    """A NarratorAgent around the real one: per-model options, and a CallRecord per call."""
    inner: NarratorAgent
    params: ModelParams
    calls: list[CallRecord] = field(default_factory=list)

    async def run(self, messages: str, *, options: dict[str, Any]) -> Any:
        start = time.perf_counter()
        try:
            response = await self.inner.run(messages, options=adapt_options(options, self.params))
        except BaseException as exc:   # a timeout cancels the call: record it, then let the Narrator handle it
            self.calls.append(CallRecord(time.perf_counter() - start, error=type(exc).__name__))
            raise
        usage = getattr(response, "usage_details", None) or {}
        self.calls.append(CallRecord(time.perf_counter() - start, usage.get("input_token_count"),
                                     usage.get("output_token_count"), usage.get("reasoning_output_token_count")))
        return response


# --- running -----------------------------------------------------------------------------------------------------

def planned_moments(events, scorer) -> int:
    """The number of commentated moments (no model call)."""
    from xplain.pipeline import analyse_match
    bundles, _ = analyse_match(events, scorer)
    return sum(b.qualifies_live for b in bundles)


async def run_model(name: str, events, scorer, credential) -> tuple[Any, list[CallRecord], float]:
    from xplain.bundle import build_bundle, narrate_match
    from xplain.config import FoundryConfig
    from xplain.narrator import build_agent

    params = MODEL_PARAMS[name]
    agent = ModelAgent(build_agent(credential, FoundryConfig(deployment=name), mode=MODE), params)
    narrator = Narrator(agent, timeout_s=params.timeout_s, mode=MODE)
    start = time.perf_counter()
    outcomes = await narrate_match(events, scorer, narrator)
    wall = time.perf_counter() - start
    return build_bundle(events, scorer, outcomes, MODE), agent.calls, wall


def bundle_path(name: str, out_dir: Path) -> Path:
    return out_dir / f"bundle_seed3001_{name}.json"


def calls_path(name: str, out_dir: Path) -> Path:
    return out_dir / f"calls_{name}.json"


async def run_all(models: list[str], match: Path, out_dir: Path) -> None:
    from azure.identity.aio import AzureCliCredential

    from xplain.bundle import _load_scorer, to_json
    from xplain.pipeline import load_events

    events = load_events(match)
    scorer = _load_scorer()
    n = planned_moments(events, scorer)
    for name in models:
        print(f"planned: {name}: {n} commentated moments, at most {n * MAX_ATTEMPTS} calls "
              f"(MAX_ATTEMPTS {MAX_ATTEMPTS})", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    async with AzureCliCredential() as credential:
        for name in models:
            print(f"running {name} ...", flush=True)
            bundle, calls, wall = await run_model(name, events, scorer, credential)
            bundle_path(name, out_dir).write_text(to_json(bundle), encoding="utf-8")
            calls_path(name, out_dir).write_text(json.dumps(
                {"model": name, "params": asdict(MODEL_PARAMS[name]), "wall_s": wall,
                 "calls": [asdict(c) for c in calls]}, indent=1), encoding="utf-8")
            print(f"  {name}: {len(calls)} calls, {wall:.1f} s wall", flush=True)


# --- report ------------------------------------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for no values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def commentated(bundle: dict) -> list[dict]:
    """The commentated moments of a bundle (parsed JSON), in seq order."""
    return sorted((m for m in bundle["moments"] if m["bundle"]["qualifies_live"]),
                  key=lambda m: m["bundle"]["moment"]["seq"])


def model_stats(name: str, bundle: dict, calls: dict) -> dict[str, Any]:
    moments = commentated(bundle)
    statuses = [m["narration"]["status"] for m in moments]
    records = calls["calls"]
    ok = [c for c in records if c["error"] is None]
    lines = len(moments)
    tok_in = sum(c["input_tokens"] or 0 for c in ok)
    tok_out = sum(c["output_tokens"] or 0 for c in ok)
    tok_reason = sum(c["reasoning_tokens"] or 0 for c in ok)
    price = PRICES.get(name)
    line_latency = [m["narration"].get("latency_s", 0.0) for m in moments]
    narrated = lv.narrated_moments(bundle)
    variety = {lang: lv.metrics([(m["moment"]["player_name"], m["line"][f"line_{lang}"]) for m in narrated], lang)
               for lang in LANGS}
    return {
        "model": name, "params": calls["params"], "moments": lines,
        "first_try": statuses.count("accepted"), "retried": statuses.count("retried"),
        "rejected": statuses.count("rejected"), "calls": len(records), "errors": len(records) - len(ok),
        "reasons": [(m["bundle"]["moment"]["minute_label"], m["bundle"]["moment"]["player_name"],
                     m["narration"]["status"], m["narration"].get("reasons", []))
                    for m in moments if m["narration"].get("reasons")],
        "call_p50": percentile([c["latency_s"] for c in ok], 50),
        "call_p95": percentile([c["latency_s"] for c in ok], 95),
        "line_p50": percentile(line_latency, 50), "line_p95": percentile(line_latency, 95),
        "in_per_line": tok_in / lines if lines else 0.0, "out_per_line": tok_out / lines if lines else 0.0,
        "reasoning_per_line": tok_reason / lines if lines else 0.0,
        "in_per_call": tok_in / len(ok) if ok else 0.0, "out_per_call": tok_out / len(ok) if ok else 0.0,
        "cost": None if price is None else (tok_in * price[0] + tok_out * price[1]) / 1e6,
        "tokens": (tok_in, tok_out), "wall_s": calls["wall_s"], "variety": variety,
    }


def _s(x: float | None, fmt: str = "{:.1f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def render(stats: list[dict], bundles: dict[str, dict]) -> str:
    names = [s["model"] for s in stats]
    out = ["# Narrator model comparison (seed3001, hints mode, measurement only)", "",
           f"Same commentated moments, prompt, frames, phrase table and Verifier for every model; MAX_ATTEMPTS "
           f"{MAX_ATTEMPTS}. Confound: gpt-6.1-sol runs at temperature 1 (it rejects any other value), with "
           f"reasoning_effort=low; the gpt-4.1 models at temperature {TEMPERATURE}.", "",
           "## Summary", "",
           "| model | params | lines | first try | retried | rejected | calls (errors) | call p50/p95 s "
           "| line p50/p95 s | tokens/line in/out (reasoning) | cost/bundle USD |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in stats:
        p = s["params"]
        params = ", ".join(f"{k}={v}" for k, v in p.items() if v is not None)
        out.append(f"| {s['model']} | {params} | {s['moments']} | {s['first_try']} | {s['retried']} | "
                   f"{s['rejected']} | {s['calls']} ({s['errors']}) | {_s(s['call_p50'])} / {_s(s['call_p95'])} | "
                   f"{_s(s['line_p50'])} / {_s(s['line_p95'])} | {s['in_per_line']:.0f} / {s['out_per_line']:.0f} "
                   f"({s['reasoning_per_line']:.0f}) | {_s(s['cost'], '{:.4f}')} |")
    out += ["", "Line latency is the Narrator's wall time for a moment (all its attempts). Tokens per line are all "
            "calls' tokens over the commentated moments (retries included). Cost uses PRICES in the script "
            "(n/a: no price set).", "", "## Repetition (scripts/line_variety.py)", "",
            "| model | lang | n | near-dup | first6 | skel-dup | mean-sim | top opener |", "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        for lang, r in s["variety"].items():
            opener, count = r["top_opener"]
            out.append(f"| {s['model']} | {lang.upper()} | {r['n']} | {r['near_dup']:.0%} | {r['first6']:.0%} | "
                       f"{r['skel_dup']:.0%} | {r['mean_sim']:.2f} | {count}/{r['n']} '{opener}' |")
    out += ["", "## Verifier reasons (retried and rejected moments)", ""]
    for s in stats:
        out.append(f"### {s['model']}")
        if not s["reasons"]:
            out.append("- none")
        for minute, player, status, reasons in s["reasons"]:
            out.append(f"- [{minute}'] {player} ({status})")
            out += [f"  - {r}" for r in reasons]
        out.append("")

    base = commentated(bundles[names[0]])
    by_model = {n: {m["bundle"]["moment"]["moment_id"]: m for m in commentated(bundles[n])} for n in names}

    def block(langs: tuple[str, ...]) -> list[str]:
        rows = []
        for slot, m in enumerate(base):
            mo = m["bundle"]["moment"]
            frame = lv.frame_for(lv.MomentBundle.model_validate(m["bundle"]), slot)
            rows.append(f"### [{mo['minute_label']}'] {mo['player_name']} | {mo['kind']} | {m['tier']['en']} | "
                        f"frame {frame}")
            for lang in langs:
                for n in names:
                    e = by_model[n].get(mo["moment_id"])
                    line = e and e.get("line")
                    text = line[f"line_{lang}"] if line else f"(no line: {e['narration']['status'] if e else 'missing'})"
                    tag = f"{n} {lang.upper()}" if len(langs) > 1 else n
                    rows.append(f"- **{tag}**: {text}")
            rows.append("")
        return rows

    out += ["## English lines side by side", ""] + block(("en",))
    out += ["## Appendix: Spanish and Greek", ""] + block(("es", "el"))
    return "\n".join(out)


def write_report(models: list[str], out_dir: Path) -> Path:
    bundles, stats = {}, []
    for name in models:
        bp, cp = bundle_path(name, out_dir), calls_path(name, out_dir)
        if not bp.exists() or not cp.exists():
            print(f"skipping {name}: no saved run in {out_dir}", file=sys.stderr)
            continue
        bundles[name] = json.loads(bp.read_text(encoding="utf-8"))
        stats.append(model_stats(name, bundles[name], json.loads(cp.read_text(encoding="utf-8"))))
    path = out_dir / "report.md"
    if stats:
        path.write_text(render(stats, bundles), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Narrate the same match with several Foundry deployments.")
    parser.add_argument("--file", type=Path, default=DEFAULT_MATCH)
    parser.add_argument("--models", nargs="+", default=list(MODEL_PARAMS), choices=list(MODEL_PARAMS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-only", action="store_true", help="no model calls: report on the saved runs")
    args = parser.parse_args(argv)
    if not args.report_only:
        asyncio.run(run_all(args.models, args.file, args.out_dir))
    print(f"report: {write_report(args.models, args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
