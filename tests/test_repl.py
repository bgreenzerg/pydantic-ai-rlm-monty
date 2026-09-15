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


def test_output_bomb_poisoned_session() -> None:
    config = RLMConfig(max_output_bytes=1024, truncate_output_chars=1024)
    repl = REPLEnvironment("x", config)
    result = repl.execute("print('x' * 4096)")

    assert not result.success
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
