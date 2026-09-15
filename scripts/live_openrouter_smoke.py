"""Run a real main-model + Monty + sub-model integration through OpenRouter."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from pydantic_ai_rlm import RLMConfig, RLMDependencies, configure_logging, create_rlm_agent


def _model_spec() -> str:
    configured = os.getenv("ASSISTANT_MODEL", "").strip()
    if not configured:
        raise RuntimeError("ASSISTANT_MODEL is missing")
    spec = configured if configured.startswith("openrouter:") else f"openrouter:{configured}"
    if "/" not in spec.partition(":")[2] or any(character.isspace() for character in spec):
        raise RuntimeError("ASSISTANT_MODEL must be an OpenRouter provider/model identifier")
    return spec


def _synthetic_context() -> list[dict[str, str]]:
    records = [
        {"id": f"evt-{index:03d}", "note": "Routine reconciliation completed with green status."}
        for index in range(1, 31)
    ]
    records[16] = {
        "id": "evt-017",
        "note": "Control AURORA-731 has amber status because the reconciliation gap is 17 units.",
    }
    return records


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

    model = _model_spec()
    config = RLMConfig(
        code_timeout=90,
        max_total_execution_seconds=180,
        max_submodel_calls=2,
        max_suspensions=4,
        sub_model=model,
    )
    deps = RLMDependencies(context=_synthetic_context(), config=config)
    agent = create_rlm_agent(model=model, sub_model=model, code_timeout=config.code_timeout)

    # Metadata is useful for proving the nested call occurred; contents remain redacted.
    logger = configure_logging(enabled=True, include_content=False)
    query = """
This is a live integration test using synthetic data. You must use execute_code at
least twice. In the first execution, search context and save the exceptional record
in a variable named target; do not call llm_query yet. In a later execution, call
llm_query with target and ask it to return the literal token LIVE_SUBMODEL_OK plus a
one-sentence interpretation. Print that sub-model response. Then answer with both
the control identifier and LIVE_SUBMODEL_OK. Do not invent either value.
""".strip()

    started = time.perf_counter()
    with (
        patch.object(logger, "log_llm_query", wraps=logger.log_llm_query) as query_log,
        patch.object(logger, "log_llm_response", wraps=logger.log_llm_response) as response_log,
    ):
        result = await agent.run(
            query,
            deps=deps,
            model_settings={"max_tokens": 2_000, "timeout": 120},
            usage_limits=UsageLimits(request_limit=10, tool_calls_limit=8, total_tokens_limit=30_000),
        )
    elapsed = time.perf_counter() - started

    messages = result.all_messages()
    tool_calls = [part for message in messages for part in message.parts if isinstance(part, ToolCallPart)]
    tool_returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]

    checks = {
        "at_least_two_execute_code_calls": sum(part.tool_name == "execute_code" for part in tool_calls) >= 2,
        "submodel_query_callback_invoked": query_log.call_count >= 1,
        "submodel_response_callback_completed": response_log.call_count >= 1,
        "final_answer_has_control": "AURORA-731" in result.output,
        "final_answer_has_submodel_marker": "LIVE_SUBMODEL_OK" in result.output,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "elapsed_seconds": round(elapsed, 3),
        "execute_code_calls": sum(part.tool_name == "execute_code" for part in tool_calls),
        "tool_results": len(tool_returns),
        "submodel_queries": query_log.call_count,
        "usage": asdict(result.usage),
        "answer": result.output,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if not report["passed"]:
        raise RuntimeError("live OpenRouter RLM smoke test failed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"Live smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
