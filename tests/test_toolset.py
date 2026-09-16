from __future__ import annotations

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_rlm import RLMConfig, RLMDependencies, SandboxFatalError, cleanup_repl_environments, create_rlm_toolset


@pytest.mark.asyncio
async def test_toolset_has_per_run_lifecycle() -> None:
    toolset = create_rlm_toolset(code_timeout=5)
    ctx = RunContext(deps=RLMDependencies(context="run-secret"), model=TestModel(), usage=RunUsage())
    run_toolset = await toolset.for_run(ctx)
    assert run_toolset is not toolset

    async with run_toolset:
        assert run_toolset._repl is None
        tools = await run_toolset.get_tools(ctx)
        tool = tools["execute_code"]
        assert tool.tool_def.sequential is True
        first = await run_toolset.call_tool("execute_code", {"code": "x = context"}, ctx, tool)
        second = await run_toolset.call_tool("execute_code", {"code": "x"}, ctx, tool)

    assert "Execution time:" in str(first.return_value)
    assert first.metadata == {"pydantic_ai_rlm": {"success": True, "output_truncated": False}}
    assert "run-secret" in str(second.return_value)
    with pytest.raises(SandboxFatalError, match="not active"):
        await run_toolset.call_tool("execute_code", {"code": "x"}, ctx, tool)


@pytest.mark.asyncio
async def test_fatal_sandbox_error_invalidates_tool_call() -> None:
    toolset = create_rlm_toolset(code_timeout=5)
    deps = RLMDependencies(
        context="run-secret",
        config=RLMConfig(max_output_bytes=512, max_emitted_output_bytes=1024, truncate_output_chars=512),
    )
    ctx = RunContext(deps=deps, model=TestModel(), usage=RunUsage())
    run_toolset = await toolset.for_run(ctx)

    async with run_toolset:
        tools = await run_toolset.get_tools(ctx)
        tool = tools["execute_code"]
        with pytest.raises(SandboxFatalError, match="output_flood"):
            await run_toolset.call_tool("execute_code", {"code": "print('x' * 4096)"}, ctx, tool)


@pytest.mark.asyncio
async def test_host_timeout_invalidates_tool_call_and_closes_worker() -> None:
    toolset = create_rlm_toolset(code_timeout=0.05)
    ctx = RunContext(deps=RLMDependencies(context="run-secret"), model=TestModel(), usage=RunUsage())
    run_toolset = await toolset.for_run(ctx)

    async with run_toolset:
        tools = await run_toolset.get_tools(ctx)
        tool = tools["execute_code"]
        with pytest.raises(SandboxFatalError, match="agent run is invalid"):
            await run_toolset.call_tool("execute_code", {"code": "while True:\n    pass"}, ctx, tool)


def test_legacy_cleanup_is_safe_noop() -> None:
    assert cleanup_repl_environments() is None


@pytest.mark.asyncio
async def test_per_run_timeout_is_not_overwritten_by_factory_default() -> None:
    toolset = create_rlm_toolset()
    deps = RLMDependencies(context="x", config=RLMConfig(code_timeout=0.05))
    ctx = RunContext(deps=deps, model=TestModel(), usage=RunUsage())
    run_toolset = await toolset.for_run(ctx)

    assert run_toolset._config.code_timeout == 0.05
    async with run_toolset:
        tools = await run_toolset.get_tools(ctx)
        with pytest.raises(SandboxFatalError, match=r"0\.05-second"):
            await run_toolset.call_tool("execute_code", {"code": "while True:\n    pass"}, ctx, tools["execute_code"])
