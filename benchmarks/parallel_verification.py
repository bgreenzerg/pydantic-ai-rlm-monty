"""Repeatable concurrency verification with real Monty workers and a deterministic model.

Run: .venv/Scripts/python.exe benchmarks/parallel_verification.py
No external LLM calls or credentials are used.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_rlm import RLMConfig, RLMDependencies, create_rlm_agent
from pydantic_ai_rlm.repl import AsyncREPLEnvironment


async def batch(count: int, hold: float) -> dict:
    active = 0
    peak = 0

    async def model(messages, info):
        nonlocal active, peak
        returns = [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": "marker = context['tenant']; print(marker); print(len(context['payload']))"},
                        tool_call_id="first",
                    )
                ]
            )
        if len(returns) == 1:
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(hold)
            finally:
                active -= 1
            return ModelResponse(
                parts=[
                    ToolCallPart("execute_code", {"code": "assert marker == context['tenant']; print(marker)"}, tool_call_id="second")
                ]
            )
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])

    agent = create_rlm_agent(model=FunctionModel(model))

    async def tenant(index):
        token = f"tenant-{index:04d}-END"
        result = await agent.run(
            "inspect",
            deps=RLMDependencies(
                context={"tenant": token, "payload": "x" * (1024 * 1024 - len(token))},
                config=RLMConfig(checkout_timeout=30),
            ),
        )
        assert token in result.output
        assert all(f"tenant-{other:04d}-END" not in result.output for other in range(count) if other != index)

    started = time.perf_counter()
    await asyncio.wait_for(asyncio.gather(*(tenant(i) for i in range(count))), timeout=90)
    return {
        "runs": count,
        "seconds": round(time.perf_counter() - started, 3),
        "peak_sessions_held_during_model_wait": peak,
        "isolation_passed": True,
    }


async def saturation() -> dict:
    holders = [AsyncREPLEnvironment(str(i), RLMConfig()) for i in range(4)]
    try:
        await asyncio.gather(*(r.open() for r in holders))
        started = time.perf_counter()
        waiting = AsyncREPLEnvironment("waiting", RLMConfig())
        try:
            await waiting.open()
            raise AssertionError("Fifth session unexpectedly acquired capacity")
        except TimeoutError:
            elapsed = time.perf_counter() - started
        finally:
            await waiting.close()
        cancelled = AsyncREPLEnvironment("cancelled", RLMConfig())
        task = asyncio.create_task(cancelled.open())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            await cancelled.close()
        outputs = await asyncio.gather(*(r.execute("print(context)") for r in holders))
        assert all(r.success and r.stdout == f"{i}\n" for i, r in enumerate(outputs))
    finally:
        await asyncio.gather(*(r.close() for r in holders))
    recovered = [AsyncREPLEnvironment(str(i), RLMConfig()) for i in range(4)]
    try:
        await asyncio.gather(*(r.open() for r in recovered))
    finally:
        await asyncio.gather(*(r.close() for r in recovered))
    return {
        "default_fifth_session_timeout_seconds": round(elapsed, 3),
        "existing_sessions_unaffected": True,
        "all_four_slots_recovered": True,
    }


async def main():
    if "--child" in sys.argv:
        print(json.dumps(await batch(4, 2)))
        return
    host = psutil.Process()
    baseline = {p.pid for p in host.children(recursive=True)}
    report = {
        "model": "deterministic FunctionModel; real Monty subprocesses",
        "context_bytes_per_run": 1024 * 1024,
        "single_process": await batch(32, 0.2),
        "saturation": await saturation(),
    }
    children = [
        await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(4)
    ]
    peak_workers = 0
    peak_rss = 0

    async def observe():
        nonlocal peak_workers, peak_rss
        while any(c.returncode is None for c in children):
            workers = []
            for p in host.children(recursive=True):
                try:
                    if "monty" in p.name().lower():
                        workers.append(p.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
            peak_workers = max(peak_workers, len(workers))
            peak_rss = max(peak_rss, sum(workers))
            await asyncio.sleep(0.02)

    observer = asyncio.create_task(observe())
    results = await asyncio.wait_for(asyncio.gather(*(c.communicate() for c in children)), 90)
    await observer
    assert all(c.returncode == 0 for c in children), results
    report["four_host_processes"] = {
        "batches": [json.loads(out) for out, _ in results],
        "peak_monty_workers": peak_workers,
        "sampled_peak_worker_rss_bytes": peak_rss,
    }
    await asyncio.sleep(0.2)
    remaining = [p.pid for p in host.children(recursive=True) if p.pid not in baseline]
    report["remaining_child_processes"] = remaining
    assert not remaining, remaining
    assert report["single_process"]["peak_sessions_held_during_model_wait"] == 4
    assert peak_workers == 16, peak_workers
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
