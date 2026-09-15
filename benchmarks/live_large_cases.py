"""Run the two imported large synthetic RLM evaluations with MLflow tracing.

The workload prompts and deterministic data generators come from the user's
previous ``pydantic-ai-rlm-harness`` evaluation project. The runner adapts them
to this repository's public ``create_rlm_agent`` API; production code is not
modified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psutil  # type: ignore[import-untyped]
from dotenv import load_dotenv
from pydantic_ai import UsageLimits
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from large_cases.accounting_data import (  # noqa: E402
    EXPECTED_SHA256 as ACCOUNTING_SHA256,
    TASK as ACCOUNTING_TASK,
    load_dataset as load_accounting_dataset,
)
from large_cases.accounting_quality import accounting_domain_quality_error  # noqa: E402
from large_cases.feedback_data import (  # noqa: E402
    EXPECTED_SHA256 as FEEDBACK_SHA256,
    TASK as FEEDBACK_TASK,
    load_dataset as load_feedback_dataset,
)
from large_cases.feedback_quality import feedback_domain_quality_error  # noqa: E402
from live_synthetic_suite import (  # noqa: E402
    DEFAULT_MLFLOW_TRACKING_URI,
    _is_monty_worker,
    configure_mlflow,
    git_commit,
    model_spec,
    monitor_memory,
)

from pydantic_ai_rlm import RLMConfig, RLMDependencies, configure_logging, create_rlm_agent  # noqa: E402

DEFAULT_EXPERIMENT = "pydantic-ai-rlm-large-benchmark"
QualityGate = Callable[[str], str | None]


@dataclass(frozen=True)
class LargeCase:
    """One imported deterministic evaluation case."""

    name: str
    description: str
    task: str
    context: str
    record_count: int
    expected_sha256: str
    quality_gate: QualityGate
    require_submodel: bool


class QueryTraceRecorder:
    """Measure and explicitly trace nested ``llm_query`` callbacks."""

    def __init__(self, mlflow: Any, root_span: Any, logger: Any, model: str) -> None:
        self._mlflow = mlflow
        self._root_span = root_span
        self._log_query_original = logger.log_llm_query
        self._log_response_original = logger.log_llm_response
        self._model = model
        self._pending: list[tuple[Any, float]] = []
        self.query_chars = 0
        self.response_chars = 0
        self.calls = 0
        self.completed = 0

    def log_query(self, prompt: str) -> None:
        """Start a child span when the sandbox invokes a submodel."""
        self._log_query_original(prompt)
        self.calls += 1
        self.query_chars += len(prompt)
        parent = self._mlflow.get_current_active_span() or self._root_span
        span = self._mlflow.start_span_no_context(
            name="rlm.llm_query",
            span_type="LLM",
            parent_span=parent,
            inputs={"prompt": prompt},
            attributes={"benchmark.model": self._model, "benchmark.prompt_chars": len(prompt)},
        )
        self._pending.append((span, time.perf_counter()))

    def log_response(self, response: str) -> None:
        """Finish the oldest pending submodel span."""
        self._log_response_original(response)
        self.response_chars += len(response)
        self.completed += 1
        if self._pending:
            span, started = self._pending.pop(0)
            span.end(
                outputs={"response": response},
                attributes={
                    "benchmark.response_chars": len(response),
                    "benchmark.elapsed_seconds": time.perf_counter() - started,
                },
                status="OK",
            )

    def close(self) -> None:
        """Close callbacks that failed before returning a response."""
        for span, started in self._pending:
            span.end(
                outputs={"error": "submodel callback did not complete"},
                attributes={"benchmark.elapsed_seconds": time.perf_counter() - started},
                status="ERROR",
            )
        self._pending.clear()


def load_cases() -> list[LargeCase]:
    """Generate and digest-check both imported JSONL corpora."""
    feedback = load_feedback_dataset()
    accounting = load_accounting_dataset()
    return [
        LargeCase(
            name="customer_feedback_5000",
            description="Find a semantically coherent checkout defect among 5,000 feedback records.",
            task=FEEDBACK_TASK,
            context=feedback,
            record_count=feedback.count("\n"),
            expected_sha256=FEEDBACK_SHA256,
            quality_gate=lambda answer: feedback_domain_quality_error(answer, record_count=feedback.count("\n")),
            require_submodel=True,
        ),
        LargeCase(
            name="accounting_3682",
            description="Answer five accounting and audit questions over 3,682 records.",
            task=ACCOUNTING_TASK,
            context=accounting,
            record_count=accounting.count("\n"),
            expected_sha256=ACCOUNTING_SHA256,
            quality_gate=lambda answer: accounting_domain_quality_error(answer, record_count=accounting.count("\n")),
            require_submodel=False,
        ),
    ]


def _case_config(model: str) -> RLMConfig:
    return RLMConfig(
        code_timeout=180,
        truncate_output_chars=10_000,
        max_context_bytes=5_000_000,
        max_output_bytes=10_000,
        max_emitted_output_bytes=4 * 1024 * 1024,
        max_total_execution_seconds=240,
        max_suspensions=12,
        max_submodel_calls=3,
        max_submodel_prompt_bytes=500_000,
        max_submodel_output_bytes=50_000,
        sub_model=model,
    )


def _tool_text(result: Any) -> tuple[int, list[str]]:
    calls = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == "execute_code"
    ]
    returns = [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == "execute_code"
    ]
    return len(calls), returns


async def _execute_case(
    case: LargeCase,
    agent: Any,
    logger: Any,
    process: psutil.Process,
    mlflow: Any,
    root_span: Any,
    api_key: str,
) -> dict[str, Any]:
    config = _case_config(model_spec())
    deps = RLMDependencies(context=case.context, config=config)
    peaks = {
        "parent": process.memory_info().rss,
        "workers": 0,
        "worker_count": 0,
        "auxiliary": 0,
        "auxiliary_count": 0,
    }
    stop = asyncio.Event()
    memory_monitor = asyncio.create_task(monitor_memory(stop, process, peaks))
    query_recorder = QueryTraceRecorder(mlflow, root_span, logger, model_spec())
    result: Any | None = None
    failure: Exception | None = None
    started = time.perf_counter()
    try:
        with (
            patch.object(logger, "log_llm_query", side_effect=query_recorder.log_query),
            patch.object(logger, "log_llm_response", side_effect=query_recorder.log_response),
        ):
            result = await agent.run(
                case.task,
                deps=deps,
                model_settings={"max_tokens": 8_000, "timeout": 240},
                usage_limits=UsageLimits(request_limit=24, tool_calls_limit=20, total_tokens_limit=250_000),
            )
    except Exception as exc:
        failure = exc
    finally:
        elapsed = time.perf_counter() - started
        query_recorder.close()
        stop.set()
        await memory_monitor

    await asyncio.sleep(0.1)
    remaining_children = process.children(recursive=True)
    remaining_workers = sum(_is_monty_worker(child) for child in remaining_children)
    base_report: dict[str, Any] = {
        "name": case.name,
        "description": case.description,
        "elapsed_seconds": round(elapsed, 3),
        "context_bytes": len(case.context.encode("utf-8")),
        "context_sha256": hashlib.sha256(case.context.encode("utf-8")).hexdigest(),
        "record_count": case.record_count,
        "nested_llm_queries": query_recorder.calls,
        "completed_nested_llm_queries": query_recorder.completed,
        "chars_sent_to_nested_llm": query_recorder.query_chars,
        "chars_received_from_nested_llm": query_recorder.response_chars,
        "selection_ratio": round(query_recorder.query_chars / len(case.context), 8),
        "peak_parent_rss_mb": round(peaks["parent"] / 1_000_000, 3),
        "peak_worker_rss_mb": round(peaks["workers"] / 1_000_000, 3),
        "peak_worker_count": peaks["worker_count"],
        "peak_auxiliary_child_rss_mb": round(peaks["auxiliary"] / 1_000_000, 3),
        "peak_auxiliary_child_count": peaks["auxiliary_count"],
        "remaining_worker_count": remaining_workers,
        "output_limits": {
            "soft_return_bytes": config.max_output_bytes,
            "hard_emitted_bytes": config.max_emitted_output_bytes,
        },
    }
    if failure is not None or result is None:
        return {
            **base_report,
            "passed": False,
            "quality_error": f"{type(failure).__name__}: {failure}" if failure is not None else "missing result",
        }

    execute_calls, tool_returns = _tool_text(result)
    tool_return_chars = sum(len(value) for value in tool_returns)
    usage = result.usage
    answer = str(result.output)
    quality_error = case.quality_gate(answer)
    trajectory_errors: list[str] = []
    if not 1 <= execute_calls <= 18:
        trajectory_errors.append(f"execute_code trajectory outside 1..18 ({execute_calls})")
    if not 2 <= usage.requests <= 20:
        trajectory_errors.append(f"main request count outside 2..20 ({usage.requests})")
    if case.require_submodel and query_recorder.completed < 1:
        trajectory_errors.append("no completed independent nested llm_query")
    maximum_selection_ratio = 0.08 if case.require_submodel else 0.15
    if query_recorder.query_chars and query_recorder.query_chars / len(case.context) > maximum_selection_ratio:
        trajectory_errors.append(
            f"nested evidence exposure exceeds {maximum_selection_ratio:.0%} ({query_recorder.query_chars / len(case.context):.3%})"
        )
    combined_output = answer + "\n".join(tool_returns)
    checks = {
        "fixture_digest": base_report["context_sha256"] == case.expected_sha256,
        "domain_quality": quality_error is None,
        "bounded_trajectory": not trajectory_errors,
        "api_key_not_disclosed": api_key not in combined_output,
        "worker_cleanup": remaining_workers == 0,
    }
    return {
        **base_report,
        "passed": all(checks.values()),
        "checks": checks,
        "quality_error": quality_error,
        "trajectory_errors": trajectory_errors,
        "execute_code_calls": execute_calls,
        "tool_execution_errors": sum("Errors:" in value or value.startswith("Error") for value in tool_returns),
        "tool_output_truncations": sum("printed output safely truncated" in value for value in tool_returns),
        "tool_return_chars": tool_return_chars,
        "largest_tool_return_chars": max(map(len, tool_returns), default=0),
        "main_agent_requests": usage.requests,
        "main_agent_input_tokens": usage.input_tokens,
        "main_agent_output_tokens": usage.output_tokens,
        "main_agent_provider_cost_usd": str(usage.cost or Decimal(0)),
        "answer": answer,
    }


async def run_case(
    case: LargeCase,
    agent: Any,
    logger: Any,
    process: psutil.Process,
    mlflow: Any,
    suite_id: str,
    api_key: str,
) -> dict[str, Any]:
    """Run a large case with a manual root and autologged nested spans."""
    attributes = {
        "benchmark.suite_id": suite_id,
        "benchmark.case": case.name,
        "benchmark.synthetic_data_only": True,
        "benchmark.model": model_spec(),
        "benchmark.git_commit": git_commit() or "unknown",
    }
    with mlflow.start_span(name=f"rlm.large.{case.name}", span_type="CHAIN", attributes=attributes) as span:
        trace_id = span.trace_id
        mlflow.set_trace_tag(trace_id, "benchmark.suite_id", suite_id)
        mlflow.set_trace_tag(trace_id, "benchmark.case", case.name)
        mlflow.set_trace_tag(trace_id, "benchmark.synthetic_data_only", "true")
        span.set_inputs(
            {
                "task": case.task,
                "context_sha256": case.expected_sha256,
                "context_bytes": len(case.context.encode("utf-8")),
                "record_count": case.record_count,
            }
        )
        report = await _execute_case(case, agent, logger, process, mlflow, span, api_key)
        report["mlflow_trace_id"] = trace_id
        span.set_outputs(
            {
                "passed": report["passed"],
                "quality_error": report.get("quality_error"),
                "trajectory_errors": report.get("trajectory_errors", []),
                "answer": report.get("answer"),
            }
        )
        span.set_attributes(
            {
                "benchmark.passed": report["passed"],
                "benchmark.elapsed_seconds": report["elapsed_seconds"],
                "benchmark.execute_code_calls": report.get("execute_code_calls", 0),
                "benchmark.nested_llm_queries": report["nested_llm_queries"],
                "benchmark.selection_ratio": report["selection_ratio"],
                "benchmark.peak_parent_rss_mb": report["peak_parent_rss_mb"],
                "benchmark.peak_worker_rss_mb": report["peak_worker_rss_mb"],
            }
        )
        if not report["passed"]:
            span.record_exception(RuntimeError(report.get("quality_error") or "benchmark quality gate failed"))

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id, silent=True, flush=True)
    trace_verified = trace is not None
    span_names = [item.name for item in trace.data.spans] if trace is not None else []
    terminal_error_spans = (
        [
            item.name
            for item in trace.data.spans
            if item.name in {"Agent.run", "OpenRouterModel.request"} and "ERROR" in str(item.status).upper()
        ]
        if trace is not None
        else []
    )
    api_key_absent_from_trace = trace is not None and api_key not in str(trace)
    trace_checks = {
        "trace_read_back": trace_verified,
        "agent_span_present": "Agent.run" in span_names,
        "tool_span_present": "execute_code" in span_names,
        "no_terminal_agent_or_provider_error": not terminal_error_spans,
        "api_key_absent_from_trace": api_key_absent_from_trace,
    }
    report["trace_checks"] = trace_checks
    report["trace_span_count"] = len(span_names)
    report["trace_span_names"] = span_names
    report["terminal_error_spans"] = terminal_error_spans
    report["passed"] = bool(report["passed"] and all(trace_checks.values()))
    mlflow.set_trace_tag(trace_id, "benchmark.passed", str(report["passed"]).lower())
    return report


async def run_suite(case_filter: str, tracking_uri: str, experiment_name: str) -> dict[str, Any]:
    """Run the selected large cases and return a reproducible JSON report."""
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or not os.getenv("ASSISTANT_MODEL"):
        raise RuntimeError("OPENROUTER_API_KEY and ASSISTANT_MODEL are required")
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    mlflow, experiment_id = configure_mlflow(tracking_uri, experiment_name)
    cases = [case for case in load_cases() if case_filter == "all" or case.name.startswith(case_filter)]
    suite_id = str(uuid.uuid4())
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    logger = configure_logging(enabled=False, include_content=False)
    agent = create_rlm_agent(model=model_spec(), sub_model=model_spec(), code_timeout=180)
    started = time.perf_counter()
    results = []
    for case in cases:
        result = await run_case(case, agent, logger, process, mlflow, suite_id, api_key)
        results.append(result)
        print(
            json.dumps(
                {
                    "progress": case.name,
                    "passed": result["passed"],
                    "seconds": result["elapsed_seconds"],
                    "trace_id": result["mlflow_trace_id"],
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    mlflow.flush_trace_async_logging()
    final_rss = process.memory_info().rss
    final_children = process.children(recursive=True)
    remaining_workers = sum(_is_monty_worker(child) for child in final_children)
    return {
        "schema_version": 1,
        "suite_passed": all(result["passed"] for result in results),
        "suite_id": suite_id,
        "case_filter": case_filter,
        "git_commit": git_commit(),
        "model": model_spec(),
        "synthetic_data_only": True,
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "experiment_id": experiment_id,
            "client_version": mlflow.__version__,
        },
        "cases_passed": sum(result["passed"] for result in results),
        "cases_total": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "main_agent_requests": sum(result.get("main_agent_requests", 0) for result in results),
        "main_agent_input_tokens": sum(result.get("main_agent_input_tokens", 0) for result in results),
        "main_agent_output_tokens": sum(result.get("main_agent_output_tokens", 0) for result in results),
        "main_agent_provider_cost_usd": str(sum(Decimal(result.get("main_agent_provider_cost_usd", "0")) for result in results)),
        "parent_rss_baseline_mb": round(baseline_rss / 1_000_000, 3),
        "parent_rss_final_mb": round(final_rss / 1_000_000, 3),
        "parent_rss_delta_mb": round((final_rss - baseline_rss) / 1_000_000, 3),
        "remaining_monty_workers": remaining_workers,
        "remaining_auxiliary_child_processes": len(final_children) - remaining_workers,
        "measurement_notes": [
            "Prompts and fixture generators are ported from the previous harness; "
            "the engine is this repository's current implementation.",
            "Provider usage and cost cover main-agent requests only; nested llm_query token usage is not aggregated.",
            "End-to-end latency and parent RSS include MLflow tracing overhead.",
            "All source data is deterministic and synthetic.",
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "customer_feedback", "accounting"), default="all")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 JSON report path")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI),
    )
    parser.add_argument(
        "--mlflow-experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT),
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(run_suite(args.case, args.mlflow_tracking_uri, args.mlflow_experiment))
    except Exception as exc:
        print(f"Large live suite failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    serialized = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Report written to {output}", file=sys.stderr)
    console_encoding = sys.stdout.encoding or "utf-8"
    console_safe = serialized.encode(console_encoding, errors="backslashreplace").decode(console_encoding)
    print(console_safe)
    if not report["suite_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
