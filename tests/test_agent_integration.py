from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelRequest, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_rlm import RLMConfig, RLMDependencies, SandboxFatalError, create_rlm_agent
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


@pytest.mark.asyncio
async def test_agent_cannot_return_normal_answer_after_fatal_sandbox_error() -> None:
    model_calls = 0

    async def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        del messages, info
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("execute_code", {"code": "print('x' * 4096)"}, tool_call_id="output-flood")]
            )
        return ModelResponse(parts=[TextPart("unsupported normal answer")])

    agent = create_rlm_agent(model=cast(Any, FunctionModel(model_function)), code_timeout=5)
    deps = RLMDependencies(
        context="classified",
        config=RLMConfig(max_output_bytes=512, max_emitted_output_bytes=1024, truncate_output_chars=512),
    )

    with pytest.raises(SandboxFatalError, match="agent run is invalid"):
        await agent.run("inspect", deps=deps)
    assert model_calls == 1


@pytest.mark.asyncio
async def test_agent_retries_internally_inconsistent_arithmetic() -> None:
    model_calls = 0

    async def model_function(messages: list[Any], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        del messages, info
        model_calls += 1
        answer = "Reported EBITDA: 14.6 + 3.5 = 17.1" if model_calls == 1 else "Reported EBITDA: 14.6 + 3.5 = 18.1"
        return ModelResponse(parts=[TextPart(answer)])

    agent = create_rlm_agent(model=cast(Any, FunctionModel(model_function)), code_timeout=5)
    result = await agent.run("calculate", deps=RLMDependencies(context="classified"))

    assert result.output == "Reported EBITDA: 14.6 + 3.5 = 18.1"
    assert model_calls == 2
