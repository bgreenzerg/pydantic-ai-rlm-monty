from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_rlm import RLMDependencies, create_rlm_agent
from pydantic_ai_rlm.prompts import build_rlm_instructions


@pytest.mark.asyncio
async def test_agent_runs_each_tenant_in_a_fresh_monty_session() -> None:
    tool_returns: list[str] = []

    async def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        # Pydantic AI strips the trailing newline while constructing a request.
        assert info.instructions == build_rlm_instructions().rstrip()
        assert [tool.name for tool in info.function_tools] == ["execute_code"]

        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_code",
                        {"code": "print(context); marker = context"},
                        tool_call_id="sandbox-call",
                    )
                ]
            )

        tool_returns.append(str(returns[-1].content))
        return ModelResponse(parts=[TextPart("complete")])

    # create_rlm_agent accepts every Pydantic AI Model at runtime; the cast only
    # bridges the fork's backwards-compatible public annotation.
    agent = create_rlm_agent(model=cast(Any, FunctionModel(model_function)), code_timeout=5)

    first = await agent.run("inspect", deps=RLMDependencies(context="tenant-a-secret"))
    second = await agent.run("inspect", deps=RLMDependencies(context="tenant-b-secret"))

    assert first.output == second.output == "complete"
    assert "tenant-a-secret" in tool_returns[0]
    assert "tenant-b-secret" not in tool_returns[0]
    assert "tenant-b-secret" in tool_returns[1]
    assert "tenant-a-secret" not in tool_returns[1]
