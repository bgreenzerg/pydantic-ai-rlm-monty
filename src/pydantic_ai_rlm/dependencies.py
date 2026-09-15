from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

ContextType = str | dict[str, Any] | list[Any]


@dataclass
class RLMConfig:
    """Configuration for RLM behavior."""

    code_timeout: float = 60.0
    """Timeout in seconds for code execution."""

    truncate_output_chars: int = 50_000
    """Maximum characters to return from code execution output."""

    sub_model: str | None = None
    """
    Model to use for llm_query() within the REPL environment.

    If set, a `llm_query(prompt: str) -> str` function becomes available
    in the REPL environment, allowing the main LLM to delegate sub-queries
    to another model. This is useful for processing large contexts in chunks.
    """

    # The limits below are deliberately finite.  A caller may choose a smaller
    # limit, but cannot turn a safety limit off with ``None`` or infinity.
    max_context_bytes: int = 16 * 1024 * 1024
    """Maximum UTF-8/JSON encoded context size accepted by a single run."""

    max_context_depth: int = 64
    """Maximum nesting depth for structured context data."""

    max_context_items: int = 1_000_000
    """Maximum number of values in structured context data."""

    max_code_bytes: int = 100_000
    """Maximum UTF-8 size of one generated code snippet."""

    max_output_bytes: int = 50_000
    """Maximum UTF-8 bytes of sandbox output retained for the main model."""

    max_emitted_output_bytes: int = 4 * 1024 * 1024
    """Hard cap on bytes emitted by one snippet before the feed is terminated."""

    validate_arithmetic: bool = True
    """Reject final answers containing internally inconsistent simple equations."""

    max_memory_bytes: int = 256 * 1024 * 1024
    """Maximum Monty worker heap allocation."""

    max_total_execution_seconds: float = 120.0
    """Maximum cumulative interpreter time across all snippets in a run."""

    max_recursion_depth: int = 256
    """Maximum Monty call-stack depth."""

    max_suspensions: int = 64
    """Maximum external calls and other suspension points per checkout."""

    max_submodel_calls: int = 16
    """Maximum ``llm_query`` calls in this run."""

    max_submodel_prompt_bytes: int = 100_000
    """Maximum UTF-8 prompt size accepted by ``llm_query``."""

    max_submodel_output_bytes: int = 50_000
    """Maximum UTF-8 response size returned from ``llm_query``."""

    checkout_timeout: float = 10.0
    """Maximum time to wait for a Monty worker checkout."""

    monty_binary_path: str | None = None
    """Optional absolute path to the trusted Monty worker binary."""

    monty_binary_sha256: str | None = None
    """Optional lowercase SHA-256 allow-list value for the worker binary."""

    def __post_init__(self) -> None:  # noqa: C901
        """Reject unbounded or otherwise invalid resource settings."""
        if not isinstance(self.validate_arithmetic, bool):
            raise TypeError("validate_arithmetic must be a boolean")
        positive_integers = {
            "truncate_output_chars": self.truncate_output_chars,
            "max_context_bytes": self.max_context_bytes,
            "max_context_depth": self.max_context_depth,
            "max_context_items": self.max_context_items,
            "max_code_bytes": self.max_code_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_emitted_output_bytes": self.max_emitted_output_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "max_recursion_depth": self.max_recursion_depth,
            "max_suspensions": self.max_suspensions,
            "max_submodel_calls": self.max_submodel_calls,
            "max_submodel_prompt_bytes": self.max_submodel_prompt_bytes,
            "max_submodel_output_bytes": self.max_submodel_output_bytes,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        finite_positive_numbers = {
            "code_timeout": self.code_timeout,
            "max_total_execution_seconds": self.max_total_execution_seconds,
            "checkout_timeout": self.checkout_timeout,
        }
        for name, numeric_value in finite_positive_numbers.items():
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(numeric_value)
                or numeric_value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")

        if self.code_timeout > 300:
            raise ValueError("code_timeout cannot exceed 300 seconds")
        if self.max_total_execution_seconds > 3600:
            raise ValueError("max_total_execution_seconds cannot exceed 3600 seconds")
        if self.checkout_timeout > 300:
            raise ValueError("checkout_timeout cannot exceed 300 seconds")
        if self.max_memory_bytes > 4 * 1024 * 1024 * 1024:
            raise ValueError("max_memory_bytes cannot exceed 4 GiB")
        if self.max_context_bytes > 1024 * 1024 * 1024:
            raise ValueError("max_context_bytes cannot exceed 1 GiB")
        if self.max_code_bytes > 10 * 1024 * 1024:
            raise ValueError("max_code_bytes cannot exceed 10 MiB")
        if self.max_output_bytes > 10 * 1024 * 1024:
            raise ValueError("max_output_bytes cannot exceed 10 MiB")
        if self.max_emitted_output_bytes > 64 * 1024 * 1024:
            raise ValueError("max_emitted_output_bytes cannot exceed 64 MiB")
        if self.max_submodel_prompt_bytes > 10 * 1024 * 1024:
            raise ValueError("max_submodel_prompt_bytes cannot exceed 10 MiB")
        if self.max_submodel_output_bytes > 10 * 1024 * 1024:
            raise ValueError("max_submodel_output_bytes cannot exceed 10 MiB")
        if self.max_context_depth > 1024:
            raise ValueError("max_context_depth cannot exceed 1024")
        if self.max_context_items > 10_000_000:
            raise ValueError("max_context_items cannot exceed 10,000,000")
        if self.max_recursion_depth > 10_000:
            raise ValueError("max_recursion_depth cannot exceed 10,000")
        if self.max_suspensions > 10_000:
            raise ValueError("max_suspensions cannot exceed 10,000")
        if self.max_submodel_calls > self.max_suspensions:
            raise ValueError("max_submodel_calls cannot exceed max_suspensions")
        if self.max_output_bytes > self.max_memory_bytes // 4:
            raise ValueError("max_output_bytes cannot exceed one quarter of max_memory_bytes")
        if self.max_emitted_output_bytes < self.max_output_bytes:
            raise ValueError("max_emitted_output_bytes cannot be smaller than max_output_bytes")
        if self.sub_model and self.max_submodel_output_bytes > self.max_memory_bytes // 4:
            raise ValueError("max_submodel_output_bytes cannot exceed one quarter of max_memory_bytes")
        if self.sub_model is not None and not isinstance(self.sub_model, str):
            raise TypeError("sub_model must be a string or None")
        if self.monty_binary_path is not None and not isinstance(self.monty_binary_path, str):
            raise TypeError("monty_binary_path must be a string or None")
        if self.monty_binary_sha256 is not None:
            digest = self.monty_binary_sha256
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("monty_binary_sha256 must be a lowercase SHA-256 hex digest")

    def context_for_monty(self, context: ContextType) -> ContextType:
        """Validate and return a host-object-free context for Monty inputs.

        Monty input conversion is intentionally limited to JSON-like values.
        In particular, arbitrary Python objects (which could retain host
        capabilities) are rejected instead of being stringified.
        """
        if not isinstance(context, (str, dict, list)):
            raise TypeError("context must be a string, dict, or list")
        if isinstance(context, str):
            size = len(context.encode("utf-8"))
            safe_context: ContextType = context
        else:
            _validate_context_value(
                context,
                max_depth=self.max_context_depth,
                max_items=self.max_context_items,
            )
            try:
                serialized = json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise ValueError("context must contain only JSON-compatible values") from exc
            size = len(serialized.encode("utf-8"))
            safe_context = json.loads(serialized)
        if size > self.max_context_bytes:
            raise ValueError(f"context exceeds max_context_bytes ({self.max_context_bytes})")
        # Monty temporarily needs roughly three copies while values cross the
        # worker boundary. Reject configurations that cannot provide headroom.
        if size > self.max_memory_bytes // 4:
            raise ValueError("encoded context cannot exceed one quarter of max_memory_bytes")
        return safe_context


@dataclass
class RLMDependencies:
    """
    Dependencies injected into RLM tools via RunContext.

    This holds the context data and configuration that
    the RLM toolset needs to operate.
    """

    context: ContextType
    """The context to analyze (string, dict, or list)."""

    config: RLMConfig = field(default_factory=RLMConfig)
    """RLM configuration options."""

    def __post_init__(self):
        """Validate dependencies after initialization."""
        if self.context is None:
            raise ValueError("context cannot be None")
        if not isinstance(self.config, RLMConfig):
            raise TypeError("config must be an RLMConfig")
        self.context = self.config.context_for_monty(self.context)


def _validate_context_value(value: Any, *, max_depth: int, max_items: int) -> None:  # noqa: C901
    """Iteratively validate a JSON tree without risking host recursion."""
    stack: list[tuple[Any, str, int]] = [(value, "context", 0)]
    seen_containers: set[int] = set()
    item_count = 0
    while stack:
        item, path, depth = stack.pop()
        item_count += 1
        if item_count > max_items:
            raise ValueError(f"context exceeds max_context_items ({max_items})")
        if item is None or isinstance(item, (str, bool, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{path} contains a non-finite float")
            continue
        if not isinstance(item, (list, dict)):
            raise TypeError(f"{path} contains unsupported value type {type(item).__name__}")
        if depth >= max_depth:
            raise ValueError(f"context exceeds max_context_depth ({max_depth})")
        identity = id(item)
        if identity in seen_containers:
            raise ValueError(f"{path} contains a cyclic or shared container")
        seen_containers.add(identity)
        if isinstance(item, list):
            stack.extend((child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item))
        else:
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} keys must be strings")
                stack.append((child, f"{path}.{key}", depth + 1))
