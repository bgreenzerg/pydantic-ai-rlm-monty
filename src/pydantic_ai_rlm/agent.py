from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, overload

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models import Model

from .dependencies import ContextType, RLMConfig, RLMDependencies
from .models import GroundedResponse
from .prompts import build_rlm_instructions
from .toolset import create_rlm_toolset
from .validation import arithmetic_consistency_errors, grounding_consistency_errors


def _has_successful_code_execution(messages: list[Any]) -> bool:
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or not isinstance(part.metadata, dict):
                continue
            metadata = part.metadata.get("pydantic_ai_rlm")
            if isinstance(metadata, dict) and metadata.get("success") is True:
                return True
    return False


@overload
def create_rlm_agent(
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    code_timeout: float | None = None,
    custom_instructions: str | None = None,
    *,
    grounded: Literal[False] = False,
) -> Agent[RLMDependencies, str]: ...


@overload
def create_rlm_agent(
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    code_timeout: float | None = None,
    custom_instructions: str | None = None,
    *,
    grounded: Literal[True],
) -> Agent[RLMDependencies, GroundedResponse]: ...


@overload
def create_rlm_agent(
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    code_timeout: float | None = None,
    custom_instructions: str | None = None,
    *,
    grounded: bool,
) -> Agent[RLMDependencies, str] | Agent[RLMDependencies, GroundedResponse]: ...


def create_rlm_agent(
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    code_timeout: float | None = None,
    custom_instructions: str | None = None,
    *,
    grounded: bool = False,
) -> Agent[RLMDependencies, str] | Agent[RLMDependencies, GroundedResponse]:
    """
    Create a Pydantic AI agent with REPL code execution capabilities.

    Args:
        model: Model to use for the main agent
        sub_model: Model to use for llm_query() within the REPL environment.
            If provided, a `llm_query(prompt: str) -> str` function becomes
            available in the REPL, allowing the agent to delegate sub-queries.
            Example: "openai:gpt-5-mini" or "anthropic:claude-3-haiku-20240307"
        code_timeout: Timeout for code execution in seconds
        custom_instructions: Additional instructions to append
        grounded: If True, return a GroundedResponse with citation markers

    Returns:
        Configured Agent instance. Returns Agent[RLMDependencies, GroundedResponse]
        when grounded=True, otherwise Agent[RLMDependencies, str].

    Example:
        ```python
        from pydantic_ai_rlm import create_rlm_agent, RLMDependencies, RLMConfig

        # Create agent with sub-model for llm_query
        agent = create_rlm_agent(
            model="openai:gpt-5",
            sub_model="openai:gpt-5-mini",
        )

        deps = RLMDependencies(
            context=very_large_document,
            config=RLMConfig(sub_model="openai:gpt-5-mini"),
        )
        result = await agent.run("What are the main themes?", deps=deps)
        print(result.output)

        # Create grounded agent
        grounded_agent = create_rlm_agent(model="openai:gpt-5", grounded=True)
        result = await grounded_agent.run("What happened?", deps=deps)
        print(result.output.info)  # Response with [N] markers
        print(result.output.grounding)  # {"1": "exact quote", ...}
        ```
    """
    toolset = create_rlm_toolset(code_timeout=code_timeout, sub_model=sub_model)

    instructions = build_rlm_instructions(
        include_llm_query=sub_model is not None,
        include_grounding=grounded,
        custom_suffix=custom_instructions,
    )

    output_type: type[str] | type[GroundedResponse] = GroundedResponse if grounded else str

    agent: Agent[RLMDependencies, Any] = Agent(
        model,
        deps_type=RLMDependencies,
        output_type=output_type,
        toolsets=[toolset],
        instructions=instructions,
    )

    @agent.output_validator
    def validate_output_arithmetic(ctx: RunContext[RLMDependencies], output: Any) -> Any:
        if ctx.deps.config.require_code_execution and not _has_successful_code_execution(ctx.messages):
            raise ModelRetry(
                "No successful execute_code result exists for this answer. Inspect the supplied context with "
                "execute_code, correct any code errors, and ground the answer in those results."
            )
        if isinstance(output, GroundedResponse):
            grounding_errors = grounding_consistency_errors(output.info, output.grounding, ctx.deps.context)
            if grounding_errors:
                raise ModelRetry(
                    "The proposed grounding is invalid: "
                    f"{'; '.join(grounding_errors)}. Use exact quotes from context with matching consecutive markers."
                )
        if not ctx.deps.config.validate_arithmetic:
            return output
        text = output.info if isinstance(output, GroundedResponse) else str(output)
        errors = arithmetic_consistency_errors(text)
        if errors:
            details = "; ".join(errors)
            raise ModelRetry(
                "The proposed final answer contains internally inconsistent arithmetic: "
                f"{details}. Recalculate every affected value with execute_code, propagate the corrections, "
                "and return a fully corrected answer."
            )
        return output

    return agent


@overload
async def run_rlm_analysis(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: Literal[False] = False,
    **agent_kwargs: Any,
) -> str: ...


@overload
async def run_rlm_analysis(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: Literal[True],
    **agent_kwargs: Any,
) -> GroundedResponse: ...


async def run_rlm_analysis(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: bool = False,
    **agent_kwargs: Any,
) -> str | GroundedResponse:
    """
    Convenience function to run RLM analysis on a context.

    Args:
        context: The large context to analyze (string, dict, or list)
        query: The question to answer about the context
        model: Model to use for the main agent
        sub_model: Model to use for llm_query() within the REPL environment.
            If provided, a `llm_query(prompt: str) -> str` function becomes
            available in the REPL, allowing the agent to delegate sub-queries.
        config: Optional RLMConfig for customization
        max_tool_calls: Maximum tool calls allowed
        grounded: If True, return a GroundedResponse with citation markers
        **agent_kwargs: Additional arguments passed to create_rlm_agent()

    Returns:
        The agent's final answer. Returns GroundedResponse when grounded=True,
        otherwise returns str.

    Example:
        ```python
        from pydantic_ai_rlm import run_rlm_analysis

        # Standard string response
        answer = await run_rlm_analysis(
            context=huge_document,
            query="Find the magic number hidden in the text",
            sub_model="openai:gpt-5-mini",
        )
        print(answer)

        # Grounded response with citations
        result = await run_rlm_analysis(
            context=document,
            query="What was the revenue change?",
            grounded=True,
        )
        print(result.info)  # "Revenue grew [1]..."
        print(result.grounding)  # {"1": "increased by 45%", ...}
        ```
    """
    effective_config = replace(config) if config is not None else RLMConfig()
    effective_sub_model = sub_model or effective_config.sub_model
    if effective_sub_model != effective_config.sub_model:
        effective_config = replace(effective_config, sub_model=effective_sub_model)
    configured_timeout: float | None = agent_kwargs.pop("code_timeout", None)
    agent = create_rlm_agent(
        model=model,
        sub_model=effective_sub_model,
        code_timeout=configured_timeout,
        grounded=grounded,
        **agent_kwargs,
    )

    deps = RLMDependencies(
        context=context,
        config=effective_config,
    )

    result = await agent.run(
        query,
        deps=deps,
        usage_limits=UsageLimits(tool_calls_limit=max_tool_calls),
    )

    return result.output


@overload
def run_rlm_analysis_sync(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: Literal[False] = False,
    **agent_kwargs: Any,
) -> str: ...


@overload
def run_rlm_analysis_sync(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: Literal[True],
    **agent_kwargs: Any,
) -> GroundedResponse: ...


def run_rlm_analysis_sync(
    context: ContextType,
    query: str,
    model: str | Model = "openai:gpt-5",
    sub_model: str | None = None,
    config: RLMConfig | None = None,
    max_tool_calls: int = 50,
    *,
    grounded: bool = False,
    **agent_kwargs: Any,
) -> str | GroundedResponse:
    """
    Synchronous version of run_rlm_analysis.

    See run_rlm_analysis() for full documentation.

    Example:
        ```python
        from pydantic_ai_rlm import run_rlm_analysis_sync

        # Standard string response
        answer = run_rlm_analysis_sync(
            context=document,
            query="What happened?",
        )

        # Grounded response with citations
        result = run_rlm_analysis_sync(
            context=document,
            query="What was the revenue change?",
            grounded=True,
        )
        print(result.info)  # "Revenue grew [1]..."
        print(result.grounding)  # {"1": "increased by 45%", ...}
        ```
    """
    effective_config = replace(config) if config is not None else RLMConfig()
    effective_sub_model = sub_model or effective_config.sub_model
    if effective_sub_model != effective_config.sub_model:
        effective_config = replace(effective_config, sub_model=effective_sub_model)
    configured_timeout: float | None = agent_kwargs.pop("code_timeout", None)
    agent = create_rlm_agent(
        model=model,
        sub_model=effective_sub_model,
        code_timeout=configured_timeout,
        grounded=grounded,
        **agent_kwargs,
    )

    deps = RLMDependencies(
        context=context,
        config=effective_config,
    )

    result = agent.run_sync(
        query,
        deps=deps,
        usage_limits=UsageLimits(tool_calls_limit=max_tool_calls),
    )

    return result.output
