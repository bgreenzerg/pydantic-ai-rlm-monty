"""Compare 4 vs 32 Monty slots in fresh processes, without changing package defaults.

Run: .venv/Scripts/python.exe benchmarks/compare_parallel_capacity.py
Results: benchmark-results/parallel-capacity.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
from parallel_verification import batch


async def trial(capacity: int, runs: int, hold: float) -> dict:
    host = psutil.Process()
    baseline = host.memory_info().rss
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            worker_rss = []
            for child in host.children(recursive=True):
                try:
                    if "monty" in child.name().lower():
                        worker_rss.append(child.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
            samples.append((len(worker_rss), sum(worker_rss), host.memory_info().rss))
            stop.wait(0.05)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        result = await batch(runs, hold)
    finally:
        stop.set()
        monitor.join(timeout=5)
    gc.collect()
    await asyncio.sleep(0.2)
    remaining = [p.pid for p in host.children(recursive=True)]
    result.update(
        {
            "capacity": capacity,
            "hold_seconds": hold,
            "host_baseline_rss_bytes": baseline,
            "host_final_rss_bytes": host.memory_info().rss,
            "peak_worker_count": max(s[0] for s in samples),
            "peak_worker_rss_bytes": max(s[1] for s in samples),
            "peak_host_rss_bytes": max(s[2] for s in samples),
            "peak_combined_rss_bytes": max(s[1] + s[2] for s in samples),
            "remaining_children": remaining,
        }
    )
    assert not remaining, result
    assert result["peak_worker_count"] == capacity, result
    assert result["peak_sessions_held_during_model_wait"] == capacity, result
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=int, choices=[4, 32])
    args = parser.parse_args()
    if args.child:
        print(json.dumps(asyncio.run(trial(args.child, 32, 2))))
        return
    results = []
    target = Path(__file__).resolve().parents[1] / "benchmark-results" / "parallel-capacity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(3):
        for capacity in [4, 32] if repetition % 2 == 0 else [32, 4]:
            started = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", str(capacity)],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1", "PYDANTIC_AI_RLM_MAX_SESSIONS": str(capacity)},
            )
            if completed.returncode:
                raise RuntimeError(completed.stdout + completed.stderr)
            result = json.loads(completed.stdout)
            result["repetition"] = repetition + 1
            result["fresh_process_wall_seconds"] = round(time.perf_counter() - started, 3)
            results.append(result)
            target.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(result), flush=True)
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
