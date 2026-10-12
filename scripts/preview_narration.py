"""
Preview the Narrator on real moments: seed 42, the first N commentated moments (qualifies_live),
narrated by the Foundry model and checked by the Verifier. Needs `az login` (AzureCliCredential).

Prints, per moment: the facts, the three lines, numbers_used, drivers_used, attempts, the Verifier
reasons and the latency; then how many were accepted first time, after the retry, or rejected.

Usage:
    python scripts/preview_narration.py [--file data/match_seed42.jsonl] [--n 8]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from azure.identity.aio import AzureCliCredential  # noqa: E402

from xplain.narrator import Narrator, build_agent, build_facts  # noqa: E402
from xplain.pipeline import ensure_match_file, load_events, process_match  # noqa: E402
from xplain.scoring import XPassScorer, find_default_model  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "match_seed42.jsonl"


async def main(path: Path, n: int) -> None:
    ensure_match_file(path, 42)
    bundles = [b for b in process_match(load_events(path), XPassScorer(*find_default_model())) if b.qualifies_live][:n]
    results = []
    async with AzureCliCredential() as credential:
        narrator = Narrator(build_agent(credential))
        for i, bundle in enumerate(bundles, start=1):
            result = await narrator.narrate(bundle)
            results.append(result)
            print(f"=== {i}. {bundle.moment.moment_id} ===")
            print("facts:", json.dumps(build_facts(bundle), ensure_ascii=False))
            if result.line is not None:
                print("EN:", result.line.line_en)
                print("ES:", result.line.line_es)
                print("EL:", result.line.line_el)
                print("numbers_used:", result.line.numbers_used, "| drivers_used:", result.line.drivers_used)
            else:
                print("no line")
            print(f"attempts: {result.attempts} | latency: {result.latency_s:.2f} s")
            print("reasons:", result.reasons or "none")
            print()

    first = sum(r.line is not None and r.attempts == 1 for r in results)
    retry = sum(r.line is not None and r.attempts > 1 for r in results)
    rejected = sum(r.line is None for r in results)
    mean = sum(r.latency_s for r in results) / len(results) if results else 0.0
    print(f"accepted first try {first}, after retry {retry}, rejected {rejected}; "
          f"mean latency {mean:.2f} s per moment")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Narrate the first commentated moments of a match.")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main(args.file, args.n))
