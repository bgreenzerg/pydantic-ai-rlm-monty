"""Compare sandbox performance for the working tree and a Git baseline.

The benchmark uses the same interpreter and dependency environment for both
revisions. It measures an active Monty lifecycle and an agent run that returns
without calling ``execute_code``. The latter exposes eager worker allocation.

Run from the repository root::

    python benchmarks/compare_release_performance.py \
        --baseline-ref d5de13d --iterations 40 \
        --output benchmark-results/release-performance.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

import psutil

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "d5de13d"


def _percentile(samples: list[float], percentage: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _is_worker(process: psutil.Process) -> bool:
    try:
        return process.name().lower() in {"monty", "monty.exe"}
    except (psutil.Error, OSError):
        return False


class _Sampler:
    """Sample parent RSS and Monty children without affecting benchmark output."""

    def __init__(self) -> None:
        self.process = psutil.Process()
        self.baseline_rss = self.process.memory_info().rss
        self.peak_rss = self.baseline_rss
        self.peak_worker_count = 0
        self.peak_worker_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.002):
            try:
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                workers = [child for child in self.process.children(recursive=True) if _is_worker(child)]
                self.peak_worker_count = max(self.peak_worker_count, len(workers))
                worker_rss = sum(child.memory_info().rss for child in workers if child.is_running())
                self.peak_worker_rss = max(self.peak_worker_rss, worker_rss)
            except (psutil.Error, OSError):
                pass

    def __enter__(self) -> _Sampler:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def report(self) -> dict[str, int]:
        return {
            "parent_peak_rss_delta_bytes": max(0, self.peak_rss - self.baseline_rss),
            "peak_worker_count": self.peak_worker_count,
            "peak_worker_rss_bytes": self.peak_worker_rss,
        }


def _latency_report(samples: list[float]) -> dict[str, float]:
    return {
        "p50_ms": round(statistics.median(samples) * 1000, 3),
        "p95_ms": round(_percentile(samples, 0.95) * 1000, 3),
        "max_ms": round(max(samples) * 1000, 3),
        "total_ms": round(sum(samples) * 1000, 3),
    }


def _active_lifecycle(iterations: int, context_bytes: int) -> dict[str, Any]:
    from pydantic_ai_rlm import REPLEnvironment, RLMConfig

    context = "x" * context_bytes
    config = RLMConfig(max_context_bytes=max(16 * 1024 * 1024, context_bytes + 1024))
    with REPLEnvironment("warmup", config) as repl:
        assert repl.execute("len(context)").success

    gc.collect()
    process = psutil.Process()
    original_children = {child.pid for child in process.children(recursive=True)}
    samples: list[float] = []
    with _Sampler() as sampler:
        for index in range(iterations):
            started = time.perf_counter()
            with REPLEnvironment(context, config) as repl:
                result = repl.execute("len(context)")
                if not result.success or result.stdout != f"{context_bytes}\n":
                    raise RuntimeError(f"active lifecycle iteration {index} returned an invalid result")
            samples.append(time.perf_counter() - started)

    gc.collect()
    time.sleep(0.1)
    leaked = {child.pid for child in process.children(recursive=True)} - original_children
    return {
        **_latency_report(samples),
        **sampler.report(),
        "leaked_child_processes": len(leaked),
    }


async def _no_tool_runs(iterations: int, context_bytes: int) -> dict[str, Any]:
    from pydantic_ai import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from pydantic_ai_rlm import RLMConfig, RLMDependencies, create_rlm_agent

    async def answer_without_tool(messages: list[Any], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[TextPart("direct answer")])

    config_kwargs: dict[str, Any] = {}
    if "require_code_execution" in RLMConfig.__dataclass_fields__:
        config_kwargs["require_code_execution"] = False
    config = RLMConfig(max_context_bytes=max(16 * 1024 * 1024, context_bytes + 1024), **config_kwargs)
    agent = create_rlm_agent(model=FunctionModel(answer_without_tool))
    context = "x" * context_bytes

    warmup = await agent.run("answer", deps=RLMDependencies(context="warmup", config=config))
    if warmup.output != "direct answer":
        raise RuntimeError("no-tool warmup returned an invalid answer")

    gc.collect()
    process = psutil.Process()
    original_children = {child.pid for child in process.children(recursive=True)}
    samples: list[float] = []
    with _Sampler() as sampler:
        for index in range(iterations):
            started = time.perf_counter()
            result = await agent.run("answer", deps=RLMDependencies(context=context, config=config))
            if result.output != "direct answer":
                raise RuntimeError(f"no-tool iteration {index} returned an invalid answer")
            samples.append(time.perf_counter() - started)

    gc.collect()
    await asyncio.sleep(0.1)
    leaked = {child.pid for child in process.children(recursive=True)} - original_children
    return {
        **_latency_report(samples),
        **sampler.report(),
        "leaked_child_processes": len(leaked),
    }


def _probe(source_root: Path, iterations: int, context_bytes: int) -> dict[str, Any]:
    sys.path.insert(0, str(source_root / "src"))
    import pydantic_ai_rlm

    return {
        "version": pydantic_ai_rlm.__version__,
        "source_root": str(source_root),
        "iterations": iterations,
        "context_bytes": context_bytes,
        "active_lifecycle": _active_lifecycle(iterations, context_bytes),
        "no_tool_agent": asyncio.run(_no_tool_runs(iterations, context_bytes)),
    }


def _run_probe(source_root: Path, iterations: int, context_bytes: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-source",
        str(source_root),
        "--iterations",
        str(iterations),
        "--context-bytes",
        str(context_bytes),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root / "src")
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _change(baseline: float, candidate: float) -> dict[str, float | str]:
    percent = 0.0 if baseline == 0 else ((candidate - baseline) / baseline) * 100
    return {
        "baseline": baseline,
        "candidate": candidate,
        "change_percent": round(percent, 2),
        "direction": "improved" if candidate < baseline else "unchanged" if candidate == baseline else "regressed",
    }


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    active_base = baseline["active_lifecycle"]
    active_candidate = candidate["active_lifecycle"]
    no_tool_base = baseline["no_tool_agent"]
    no_tool_candidate = candidate["no_tool_agent"]
    active_threshold = max(active_base["p95_ms"] * 1.25, active_base["p95_ms"] + 2.0)
    checks = {
        "no_worker_leaks": active_candidate["leaked_child_processes"] == 0 and no_tool_candidate["leaked_child_processes"] == 0,
        "active_lifecycle_p95_within_25_percent": active_candidate["p95_ms"] <= active_threshold,
        "no_tool_worker_eliminated": no_tool_candidate["peak_worker_count"] == 0,
        "no_tool_p50_improved": no_tool_candidate["p50_ms"] < no_tool_base["p50_ms"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "active_lifecycle_p50_ms": _change(active_base["p50_ms"], active_candidate["p50_ms"]),
        "active_lifecycle_p95_ms": _change(active_base["p95_ms"], active_candidate["p95_ms"]),
        "no_tool_p50_ms": _change(no_tool_base["p50_ms"], no_tool_candidate["p50_ms"]),
        "no_tool_total_ms": _change(no_tool_base["total_ms"], no_tool_candidate["total_ms"]),
        "no_tool_peak_workers": _change(no_tool_base["peak_worker_count"], no_tool_candidate["peak_worker_count"]),
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise RuntimeError("Git archive contains an unsafe path")
        bundle.extractall(destination)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--context-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations <= 0 or args.context_bytes <= 0:
        parser.error("--iterations and --context-bytes must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.probe_source is not None:
        print(json.dumps(_probe(args.probe_source.resolve(), args.iterations, args.context_bytes)))
        return

    baseline_commit = _git("rev-parse", "--verify", f"{args.baseline_ref}^{{commit}}")
    candidate_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    with tempfile.TemporaryDirectory(prefix="pydantic-ai-rlm-perf-") as temporary:
        temp_root = Path(temporary)
        archive = temp_root / "baseline.zip"
        baseline_root = temp_root / "baseline"
        baseline_root.mkdir()
        _git("archive", "--format=zip", f"--output={archive}", baseline_commit)
        _safe_extract(archive, baseline_root)
        baseline = _run_probe(baseline_root, args.iterations, args.context_bytes)
        candidate = _run_probe(REPOSITORY_ROOT, args.iterations, args.context_bytes)

    report = {
        "schema_version": 1,
        "baseline_ref": args.baseline_ref,
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "candidate_worktree_dirty": dirty,
        "python": sys.version,
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _comparison(baseline, candidate),
        "notes": [
            "Both revisions use the same Python interpreter, dependencies and machine.",
            "Active lifecycle includes worker startup, context transfer, one execution and teardown.",
            "No-tool latency measures agent runs whose model returns without execute_code.",
            "RSS sampling is approximate and is not used as a pass/fail regression gate.",
        ],
    }
    if args.output is not None:
        _write_report(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["comparison"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
