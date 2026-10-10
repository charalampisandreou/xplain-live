"""
Replay a generated match against the scoring API.

    python -m xplain.replay --file data/match_seed42.jsonl --url http://127.0.0.1:8000 --speed 0

Pass events are POSTed to {url}/score one by one; other events are counted, not sent.
--speed S waits (t_next - t_prev) / S seconds between events: 1 = real time,
60 = one match minute per second, 0 = no waiting. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

BANDS = ("routine", "moderate", "hard", "exceptional")
OUTLIER_MARKERS = {"surprise_success": "<< SURPRISE SUCCESS", "surprise_failure": "<< SURPRISE FAILURE"}


class ServerUnreachable(RuntimeError):
    pass


def mmss(t: float) -> str:
    total = int(t)
    return f"{total // 60:02d}:{total % 60:02d}"


def _request(url: str, data: bytes | None = None, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ServerUnreachable(f"Cannot reach the scoring API at {url} ({reason}). "
                                f"Is it running? Start it with: uvicorn xplain.api:app --port 8000") from exc


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


def replay(file: Path, url: str, speed: float, out=sys.stdout) -> dict:
    base = url.rstrip("/")
    health = _request(f"{base}/health")
    print(f"scoring API {base}: model {health['model_version']} ({health['n_features']} features)", file=out)

    lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: Counter = Counter()
    scored: list[dict] = []
    latencies_ms: list[float] = []
    prev_t: float | None = None

    for line in lines:
        event = json.loads(line)
        if speed > 0 and prev_t is not None and event["t"] > prev_t:
            time.sleep((event["t"] - prev_t) / speed)
        prev_t = event["t"]
        counts[event["type"]] += 1
        if event["type"] != "pass":
            continue

        start = time.perf_counter()
        res = _request(f"{base}/score", data=line.encode("utf-8"))
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        res["team_id"], res["player_id"] = event["team_id"], event["player_id"]
        scored.append(res)
        print(
            f"{res['seq']:5d} {mmss(res['t'])} {res['team_id']:<4} {res['player_id']:<9} "
            f"xP {res['xp']:.3f} diff {res['difficulty']:3d} {res['difficulty_band']:<11} "
            f"{'completed' if res['completed'] else 'failed   '} delta {res['delta']:+.3f} "
            f"{OUTLIER_MARKERS.get(res['outlier'], '')}".rstrip(),
            file=out,
        )

    summary = summarize(scored, latencies_ms, counts)
    print_summary(summary, out)
    return summary


def summarize(scored: list[dict], latencies_ms: list[float], counts: Counter) -> dict:
    n = len(scored)
    outliers = Counter(r["outlier"] for r in scored if r["outlier"])
    flagged = sorted((r for r in scored if r["outlier"]), key=lambda r: (-abs(r["delta"]), r["seq"]))
    return {
        "events": dict(counts),
        "passes_scored": n,
        "mean_xp": statistics.fmean(r["xp"] for r in scored) if n else float("nan"),
        "completion_rate": sum(r["completed"] for r in scored) / n if n else float("nan"),
        "mean_difficulty": statistics.fmean(r["difficulty"] for r in scored) if n else float("nan"),
        "bands": {b: sum(r["difficulty_band"] == b for r in scored) for b in BANDS},
        "outliers": {k: outliers.get(k, 0) for k in OUTLIER_MARKERS},
        "latency_median_ms": statistics.median(latencies_ms) if latencies_ms else float("nan"),
        "latency_p95_ms": p95(latencies_ms) if latencies_ms else float("nan"),
        "top_flagged": flagged[:10],
        "model_version": scored[0]["model_version"] if scored else None,
    }


def print_summary(s: dict, out=sys.stdout) -> None:
    n = s["passes_scored"]
    print("\n=== Summary ===", file=out)
    print(f"model                {s['model_version']}", file=out)
    print("events               " + ", ".join(f"{k} {v}" for k, v in sorted(s["events"].items())), file=out)
    print(f"passes scored        {n}", file=out)
    print(f"mean xP              {s['mean_xp']:.3f}", file=out)
    print(f"actual completion    {s['completion_rate']:.3f}", file=out)
    print(f"mean difficulty      {s['mean_difficulty']:.1f}", file=out)
    print("bands                " + ", ".join(f"{b} {c} ({c / n:.1%})" for b, c in s["bands"].items()), file=out)
    print("outliers             " + ", ".join(f"{k} {c} ({c / n:.1%})" for k, c in s["outliers"].items()),
          file=out)
    print(f"latency              median {s['latency_median_ms']:.1f} ms, p95 {s['latency_p95_ms']:.1f} ms",
          file=out)
    print("\nlargest |delta| flagged moments:", file=out)
    for r in s["top_flagged"]:
        print(f"  {mmss(r['t'])}  {r['player_id']:<9} xP {r['xp']:.3f}  "
              f"{'completed' if r['completed'] else 'failed   '}  delta {r['delta']:+.3f}  {r['outlier']}",
              file=out)
    if not s["top_flagged"]:
        print("  (none)", file=out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a match JSONL against the scoring API.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--speed", type=float, default=0.0, help="1 = real time, 60 = 1 match minute/s, 0 = no wait")
    args = parser.parse_args(argv)
    if args.speed < 0:
        parser.error("--speed must be >= 0")
    if not args.file.is_file():
        parser.error(f"file not found: {args.file}")
    try:
        replay(args.file, args.url, args.speed)
    except ServerUnreachable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
