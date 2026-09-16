from __future__ import annotations

import asyncio
import hashlib

import pytest

from pydantic_ai_rlm import REPLEnvironment, RLMConfig
from pydantic_ai_rlm.repl import AsyncREPLEnvironment, _trusted_binary


def test_sync_state_and_context_persist() -> None:
    with REPLEnvironment("tenant-a-secret") as repl:
        assert repl.execute("saved = context").success
        assert repl._context is None
        result = repl.execute("saved")

    assert result.success
    assert result.stdout == "'tenant-a-secret'\n"
    assert result.locals == {}


def test_prompt_documented_analysis_imports_work() -> None:
    with REPLEnvironment({"value": "abc-123"}) as repl:
        result = repl.execute(
            "import collections, json, re\n"
            "counter = collections.Counter(re.findall(r'[a-z]+', context['value']))\n"
            "print(json.dumps(dict(counter)))"
        )

    assert result.success
    assert result.stdout == '{"abc": 1}\n'


@pytest.mark.parametrize(
    "code, expected",
    [
        ("import subprocess", "ModuleNotFoundError"),
        ("import socket", "ModuleNotFoundError"),
        ("open('/etc/passwd').read()", "PermissionError"),
        ("import os\nos.getenv('PATH')", "not supported"),
        ("__import__('os')", "NameError"),
        ("globals()", "NameError"),
        ("eval('1 + 1')", "NameError"),
        ("compile('1 + 1', 'x', 'eval')", "NameError"),
        ("().__class__", "AttributeError"),
    ],
)
def test_host_capabilities_are_denied(code: str, expected: str) -> None:
    with REPLEnvironment("classified") as repl:
        result = repl.execute(code)

    assert not result.success
    assert expected in result.stderr
    assert "classified" not in result.stderr


def test_code_size_is_bounded() -> None:
    config = RLMConfig(max_code_bytes=8)
    with REPLEnvironment("x", config) as repl, pytest.raises(ValueError, match="max_code_bytes"):
        repl.execute("print('too large')")


def test_large_output_is_truncated_without_losing_session_state() -> None:
    config = RLMConfig(max_output_bytes=1024, max_emitted_output_bytes=8192, truncate_output_chars=1024)
    repl = REPLEnvironment("x", config)
    result = repl.execute("saved = 42\nprint('x' * 4096)")

    assert result.success
    assert result.output_truncated
    assert result.emitted_output_bytes == 4097
    assert "printed output safely truncated" in result.stdout
    assert len(result.stdout.encode("utf-8")) <= config.max_output_bytes
    assert repl.execute("saved + 1").stdout == "43\n"
    repl.close()


def test_large_final_expression_is_bounded_before_host_return() -> None:
    config = RLMConfig(max_output_bytes=1024, max_emitted_output_bytes=8192, truncate_output_chars=1024)
    with REPLEnvironment("x", config) as repl:
        result = repl.execute("'x' * 4096")

    assert result.success
    assert result.output_truncated
    assert len(result.stdout.encode("utf-8")) <= config.max_output_bytes


def test_large_exception_is_bounded_and_session_remains_usable() -> None:
    config = RLMConfig(max_output_bytes=1024, max_emitted_output_bytes=8192, truncate_output_chars=1024)
    with REPLEnvironment("x", config) as repl:
        result = repl.execute("raise ValueError('x' * 100_000)")
        follow_up = repl.execute("40 + 2")

    assert not result.success
    assert not result.fatal
    assert len(result.stderr.encode("utf-8")) <= config.max_output_bytes
    assert follow_up.stdout == "42\n"


def test_syntax_error_does_not_discard_first_context_feed() -> None:
    with REPLEnvironment("still-available") as repl:
        invalid = repl.execute("if")
        valid = repl.execute("context")

    assert not invalid.success
    assert invalid.failure_kind == "code_error"
    assert valid.stdout == "'still-available'\n"


def test_oversized_llm_query_prompt_is_rejected_inside_sandbox() -> None:
    config = RLMConfig(sub_model="openai:not-called", max_submodel_prompt_bytes=32)
    with REPLEnvironment("x", config) as repl:
        result = repl.execute("llm_query('x' * 1000)")

    assert not result.success
    assert "configured limit" in result.stderr
    assert repl._submodel_calls == 0


def test_sync_llm_query_survives_sandbox_exception_wrapper() -> None:
    calls: list[str] = []
    with REPLEnvironment("x", RLMConfig(sub_model="fake:model")) as repl:
        repl._llm_query = lambda prompt: calls.append(prompt) or "sync-ok"  # type: ignore[method-assign]
        result = repl.execute("llm_query('bounded evidence')")

    assert result.success
    assert result.stdout == "'sync-ok'\n"
    assert calls == ["bounded evidence"]


@pytest.mark.asyncio
async def test_async_llm_query_survives_sandbox_exception_wrapper() -> None:
    calls: list[str] = []

    async def fake_query(prompt: str) -> str:
        await asyncio.sleep(0)
        calls.append(prompt)
        return "async-ok"

    async with AsyncREPLEnvironment("x", RLMConfig(sub_model="fake:model")) as repl:
        repl._llm_query = fake_query  # type: ignore[method-assign]
        result = await repl.execute("await llm_query('bounded evidence')")

    assert result.success
    assert result.stdout == "'async-ok'\n"
    assert calls == ["bounded evidence"]


def test_output_flood_is_fatal_only_at_hard_limit() -> None:
    config = RLMConfig(max_output_bytes=1024, max_emitted_output_bytes=2048, truncate_output_chars=1024)
    repl = REPLEnvironment("x", config)
    result = repl.execute("print('x' * 4096)")

    assert not result.success
    assert result.fatal
    assert result.failure_kind == "output_flood"
    assert "MemoryError" in result.stderr
    with pytest.raises(RuntimeError, match="closed or unusable"):
        repl.execute("1 + 1")


def test_worker_binary_can_be_pinned_by_digest() -> None:
    path = _trusted_binary(RLMConfig())
    with open(path, "rb") as worker:
        digest = hashlib.file_digest(worker, "sha256").hexdigest()

    with REPLEnvironment("x", RLMConfig(monty_binary_path=path, monty_binary_sha256=digest)) as repl:
        assert repl.execute("1 + 1").success

    with pytest.raises(PermissionError, match="SHA-256"):
        REPLEnvironment("x", RLMConfig(monty_binary_path=path, monty_binary_sha256="0" * 64))


def test_worker_binary_override_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        REPLEnvironment("x", RLMConfig(monty_binary_path="monty.exe"))


def test_timeout_poisoned_session() -> None:
    config = RLMConfig(code_timeout=2, max_total_execution_seconds=0.02)
    repl = REPLEnvironment("x", config)
    result = repl.execute("while True:\n    pass")

    assert not result.success
    assert result.fatal
    assert result.failure_kind == "execution_timeout"
    assert "TimeoutError" in result.stderr
    with pytest.raises(RuntimeError, match="closed or unusable"):
        repl.execute("1 + 1")


@pytest.mark.asyncio
async def test_concurrent_tenants_are_isolated() -> None:
    async def run_tenant(index: int) -> str:
        secret = f"tenant-{index}-sentinel"
        async with AsyncREPLEnvironment(secret, RLMConfig()) as repl:
            assert (await repl.execute("saved = context")).success
            await asyncio.sleep(0)
            return (await repl.execute("saved")).stdout

    outputs = await asyncio.gather(*(run_tenant(index) for index in range(6)))

    for index, output in enumerate(outputs):
        assert output == repr(f"tenant-{index}-sentinel") + "\n"
        for other in range(6):
            if other != index:
                assert f"tenant-{other}-sentinel" not in output


@pytest.mark.asyncio
async def test_async_large_output_is_truncated_without_losing_session_state() -> None:
    config = RLMConfig(max_output_bytes=1024, max_emitted_output_bytes=8192, truncate_output_chars=1024)
    async with AsyncREPLEnvironment("x", config) as repl:
        result = await repl.execute("saved = 42\nprint('x' * 4096)")

        assert result.success
        assert result.output_truncated
        assert "printed output safely truncated" in result.stdout
        assert (await repl.execute("saved + 1")).stdout == "43\n"


@pytest.mark.asyncio
async def test_cancelled_execution_can_be_closed_without_leaking_worker() -> None:
    repl = await AsyncREPLEnvironment("cancel-secret", RLMConfig(code_timeout=5)).open()
    task = asyncio.create_task(repl.execute("while True:\n    pass"))
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await repl.close()

    assert repl._context is None
    assert repl._session is None
    assert repl._pool is None
    assert not repl._gate_acquired


@pytest.mark.asyncio
async def test_cancelled_startup_releases_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = asyncio.Event()

    class SlowPool:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def __aenter__(self) -> SlowPool:
            await asyncio.sleep(60)
            return self

        async def __aexit__(self, *args: object) -> None:
            del args
            closed.set()

    monkeypatch.setattr("pydantic_ai_rlm.repl.AsyncMonty", SlowPool)
    repl = AsyncREPLEnvironment("startup-secret", RLMConfig(code_timeout=5))
    task = asyncio.create_task(repl.open())
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed.is_set()
    assert not repl._gate_acquired
    assert repl._context is None
    await repl.close()
    assert repl._context is None
