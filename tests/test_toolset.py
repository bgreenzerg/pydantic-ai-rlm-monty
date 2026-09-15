from __future__ import annotations

import pytest
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_rlm import RLMDependencies, cleanup_repl_environments, create_rlm_toolset


@pytest.mark.asyncio
async def test_toolset_has_per_run_lifecycle() -> None:
    toolset = create_rlm_toolset(code_timeout=5)
    ctx = RunContext(deps=RLMDependencies(context="run-secret"), model=TestModel(), usage=RunUsage())
    run_toolset = await toolset.for_run(ctx)
    assert run_toolset is not toolset

    async with run_toolset:
        tools = await run_toolset.get_tools(ctx)
        tool = tools["execute_code"]
        assert tool.tool_def.sequential is True
        first = await run_toolset.call_tool("execute_code", {"code": "x = context"}, ctx, tool)
        second = await run_toolset.call_tool("execute_code", {"code": "x"}, ctx, tool)

    assert "Execution time:" in first
    assert "run-secret" in second
    after_close = await run_toolset.call_tool("execute_code", {"code": "x"}, ctx, tool)
    assert "not active" in after_close


def test_legacy_cleanup_is_safe_noop() -> None:
    assert cleanup_repl_environments() is None
