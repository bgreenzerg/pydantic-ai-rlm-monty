"""Measure Monty lifecycle latency and detect obvious worker/RSS leaks.

Run from the repository root with::

    python benchmarks/sandbox_lifecycle.py --iterations 100
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from pydantic_ai_rlm import REPLEnvironment, RLMConfig  # noqa: E402


def percentile(samples: list[float], percentage: float) -> float:
    """Return an interpolated percentile for a non-empty sample."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run(iterations: int, context_bytes: int) -> dict[str, float | int]:
    """Run isolated sessions and report latency plus retained resources."""
    process = psutil.Process()
    context = "x" * context_bytes
    config = RLMConfig(max_context_bytes=max(16 * 1024 * 1024, context_bytes + 1024))

    # Warm imports and native runtime initialization before the leak baseline.
    with REPLEnvironment("warmup", config) as repl:
        assert repl.execute("len(context)").success
    gc.collect()
    baseline_rss = process.memory_info().rss
    baseline_children = {child.pid for child in process.children(recursive=True)}

    latencies: list[float] = []
    for index in range(iterations):
        started = time.perf_counter()
        with REPLEnvironment(context, config) as repl:
            result = repl.execute("len(context)")
            if not result.success or result.stdout != f"{context_bytes}\n":
                raise RuntimeError(f"iteration {index} returned an invalid result")
        latencies.append(time.perf_counter() - started)

    gc.collect()
    time.sleep(0.2)
    final_rss = process.memory_info().rss
    final_children = {child.pid for child in process.children(recursive=True)}
    leaked_children = final_children - baseline_children

    report: dict[str, float | int] = {
        "iterations": iterations,
        "context_bytes": context_bytes,
        "latency_p50_ms": round(statistics.median(latencies) * 1000, 3),
        "latency_p95_ms": round(percentile(latencies, 0.95) * 1000, 3),
        "latency_max_ms": round(max(latencies) * 1000, 3),
        "parent_rss_baseline_bytes": baseline_rss,
        "parent_rss_final_bytes": final_rss,
        "parent_rss_delta_bytes": final_rss - baseline_rss,
        "leaked_child_processes": len(leaked_children),
    }
    if leaked_children:
        raise RuntimeError(f"worker processes leaked: {sorted(leaked_children)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--context-bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()
    if args.iterations <= 0 or args.context_bytes <= 0:
        parser.error("arguments must be positive")
    print(json.dumps(run(args.iterations, args.context_bytes), indent=2))


if __name__ == "__main__":
    main()
