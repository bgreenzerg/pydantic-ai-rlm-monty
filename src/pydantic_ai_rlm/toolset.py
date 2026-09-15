from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from .dependencies import RLMConfig, RLMDependencies
from .logging import get_logger
from .repl import AsyncREPLEnvironment, REPLResult
from .utils import format_repl_result

EXECUTE_CODE_DESCRIPTION = """
Execute Python code in a sandboxed REPL environment.

## Environment
- A `context` variable is pre-loaded with the data to analyze
- Variables persist between executions within the same session
- Standard library modules are available (json, re, collections, etc.)
- Use print() only for compact results that you need to inspect
- Printed output is safely truncated at a configured limit without losing REPL state

## When to Use
- Analyzing or processing structured data (JSON, dicts, lists)
- Performing calculations or data transformations
- Extracting specific information from large datasets
- Testing hypotheses about the data structure

## Best Practices
1. Start by exploring the context: `print(type(context))`, `print(len(context))`
2. Break complex operations into smaller steps
3. Never print an entire dataset or collection; print counts, aggregates, top-k rows, IDs, or bounded excerpts
4. Prefer `sum(1 for item in values if predicate)` over summing booleans for Monty compatibility
5. If output is truncated, reuse persistent variables and issue a smaller, more selective query
6. Handle ordinary code errors gracefully with a corrected snippet

## Available Functions
- `llm_query(prompt)`: Query the LLM for reasoning assistance (if configured)
- Important: Do not use `llm_query` in the first code execution. Use it only after you have
  explored the context and identified specific sections that need semantic analysis.
- If you claim an independent semantic review, you must actually call `llm_query` on the
  bounded evidence subset and reconcile its result with deterministic counts.

## Example
```python
# Explore the data
print(f"Context type: {type(context)}")
print(f"Keys: {list(context.keys()) if isinstance(context, dict) else 'N/A'}")

# Process and extract information
if isinstance(context, dict):
    for key, value in context.items():
        print(f"{key}: {type(value)}")
```
"""


class SandboxFatalError(RuntimeError):
    """A fatal sandbox condition that invalidates the complete agent run."""


def create_rlm_toolset(
    *,
    code_timeout: float = 60.0,
    sub_model: str | None = None,
    toolset_id: str | None = None,
) -> FunctionToolset[RLMDependencies]:
    """Create an RLM toolset for code execution in a sandboxed REPL.

    This toolset provides an `execute_code` tool that allows AI agents to
    run Python code with access to a `context` variable containing data to analyze.

    Args:
        code_timeout: Timeout in seconds for code execution. Defaults to 60.0.
        sub_model: Model to use for llm_query() within the REPL environment.
        toolset_id: Optional unique identifier for the toolset.

    Returns:
        FunctionToolset compatible with any pydantic-ai agent.

    Example (basic usage):
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_rlm import create_rlm_toolset, RLMDependencies

        toolset = create_rlm_toolset()
        agent = Agent("openai:gpt-5", toolsets=[toolset])

        deps = RLMDependencies(context={"users": [...]})
        result = await agent.run("Analyze the user data", deps=deps)
        ```

    Example (with timeout and sub-model):
        ```python
        from pydantic_ai_rlm import create_rlm_toolset, RLMDependencies, RLMConfig

        toolset = create_rlm_toolset(
            code_timeout=120.0,
            sub_model="openai:gpt-5-mini",
        )
        agent = Agent("openai:gpt-5", toolsets=[toolset])

        deps = RLMDependencies(
            context=large_dataset,
            config=RLMConfig(),
        )
        result = await agent.run("Process this dataset", deps=deps)
        ```

    Example (with toolset composition):
        ```python
        from pydantic_ai_rlm import create_rlm_toolset

        rlm_toolset = create_rlm_toolset().prefixed("rlm")
        # Tool will be named 'rlm_execute_code'
        ```
    """
    if code_timeout <= 0 or code_timeout > 300:
        raise ValueError("code_timeout must be greater than zero and at most 300 seconds")
    return MontyRLMToolset(
        code_timeout=code_timeout,
        sub_model=sub_model,
        toolset_id=toolset_id,
    )


class MontyRLMToolset(FunctionToolset[RLMDependencies]):
    """A stateless toolset factory that creates one sandbox per agent run."""

    def __init__(
        self,
        *,
        code_timeout: float,
        sub_model: str | None,
        toolset_id: str | None,
    ) -> None:
        super().__init__(id=toolset_id)
        self._code_timeout = code_timeout
        self._sub_model = sub_model

    async def for_run(self, ctx: RunContext[RLMDependencies]) -> FunctionToolset[RLMDependencies]:
        config = replace(ctx.deps.config, code_timeout=self._code_timeout)
        if self._sub_model and not config.sub_model:
            config = replace(config, sub_model=self._sub_model)
        return _RunMontyRLMToolset(
            context=ctx.deps.context,
            config=config,
            toolset_id=self.id,
        )


class _RunMontyRLMToolset(FunctionToolset[RLMDependencies]):
    """Per-run toolset owning a single isolated Monty worker session."""

    def __init__(self, *, context: Any, config: RLMConfig, toolset_id: str | None) -> None:
        super().__init__(id=toolset_id)
        self._context = context
        self._config = config
        self._repl: AsyncREPLEnvironment | None = None

        @self.tool(description=EXECUTE_CODE_DESCRIPTION, sequential=True)
        async def execute_code(ctx: RunContext[RLMDependencies], code: str) -> str:
            del ctx
            return await self._execute_code(code)

    async def __aenter__(self) -> _RunMontyRLMToolset:
        if self._repl is not None:
            raise RuntimeError("sandbox session is already active")
        context = self._context
        if context is None:
            raise SandboxFatalError("sandbox context is unavailable; agent run is invalid")
        repl = AsyncREPLEnvironment(context, self._config)
        try:
            await repl.open()
        except BaseException:
            # Do not retain tenant data when startup fails or is cancelled.
            self._context = None
            await repl.close()
            raise
        self._repl = repl
        # The validated context now belongs only to the environment/session.
        self._context = None
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        repl, self._repl = self._repl, None
        if repl is not None:
            await repl.close()
        self._context = None
        return None

    async def _execute_code(self, code: str) -> str:
        repl = self._repl
        if repl is None:
            raise SandboxFatalError("sandbox session is not active; agent run is invalid")
        logger = get_logger()
        logger.log_code_execution(code)
        try:
            result: REPLResult = await asyncio.wait_for(repl.execute(code), timeout=self._config.code_timeout)
        except TimeoutError:
            await repl.close()
            self._repl = None
            raise SandboxFatalError(
                f"sandbox execution exceeded the {self._config.code_timeout:g}-second host deadline; agent run is invalid"
            ) from None
        except asyncio.CancelledError:
            try:
                await repl.close()
            finally:
                self._repl = None
            raise
        except Exception as exc:
            await repl.close()
            self._repl = None
            raise SandboxFatalError("sandbox execution failed internally; agent run is invalid") from exc

        logger.log_result(result)
        if result.fatal:
            await repl.close()
            self._repl = None
            kind = result.failure_kind or "unknown"
            raise SandboxFatalError(f"sandbox terminated ({kind}); agent run is invalid")
        return format_repl_result(result)


def cleanup_repl_environments() -> None:
    """Compatibility no-op; environments now clean up per agent run."""
