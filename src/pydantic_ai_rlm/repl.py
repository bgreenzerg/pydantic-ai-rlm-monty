from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import os
import sys
import textwrap
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request, model_request_sync
from pydantic_ai.messages import TextPart
from pydantic_ai.settings import ModelSettings
from pydantic_monty import (
    AsyncMonty,
    AsyncMontySession,
    CollectStreams,
    Monty,
    MontyCrashedError,
    MontyError,
    MontyRuntimeError,
    MontySession,
    ResourceLimits,
)

from .dependencies import ContextType, RLMConfig

# A process-wide defence-in-depth ceiling. Each sandbox can consume up to the
# configured memory budget, so unbounded pool creation is not acceptable.
_SANDBOX_GATE = threading.BoundedSemaphore(4)


@dataclass
class REPLResult:
    """Result from sandboxed code execution."""

    stdout: str
    """Standard output plus the final expression value, if any."""

    stderr: str
    """Standard error and a sanitized interpreter error, if any."""

    locals: dict[str, Any]
    """Always empty; Monty deliberately does not expose namespace introspection."""

    execution_time: float
    """Wall-clock time spent executing the feed."""

    success: bool = True
    """Whether execution completed without an interpreter error."""

    output_truncated: bool = False
    """Whether printed output was safely shortened before returning it to the model."""

    emitted_output_bytes: int = 0
    """Total UTF-8 bytes printed by the snippet before any soft truncation."""

    fatal: bool = False
    """Whether the sandbox can no longer be trusted for this agent run."""

    failure_kind: str | None = None
    """Stable, content-free failure category for controller decisions and telemetry."""

    def __str__(self) -> str:
        return f"REPLResult(success={self.success}, stdout={self.stdout[:100]}..., stderr={self.stderr[:100]}...)"


def _limits(config: RLMConfig) -> ResourceLimits:
    return {
        "max_memory": config.max_memory_bytes,
        "max_duration_secs": config.max_total_execution_seconds,
        "max_recursion_depth": config.max_recursion_depth,
        "max_suspensions": config.max_suspensions,
    }


def _submodel_settings(config: RLMConfig) -> ModelSettings:
    """Bound provider wait time and response tokens as well as response bytes."""
    return {
        "timeout": config.code_timeout,
        "max_tokens": max(1, min(64_000, config.max_submodel_output_bytes // 4)),
    }


def _packaged_worker() -> tuple[Path, str]:
    """Resolve the worker strictly through its installed, hashed manifest entry."""
    binary_name = "monty.exe" if sys.platform == "win32" else "monty"
    try:
        runtime = distribution("pydantic-monty-runtime")
    except PackageNotFoundError as exc:
        raise FileNotFoundError("the pydantic-monty-runtime distribution is not installed") from exc
    matches = [entry for entry in runtime.files or () if entry.name == binary_name]
    if len(matches) != 1:
        raise FileNotFoundError("the Monty runtime manifest does not identify exactly one worker binary")
    entry = matches[0]
    if entry.hash is None or entry.hash.mode != "sha256":
        raise PermissionError("the Monty runtime manifest does not contain a SHA-256 worker digest")
    return Path(str(runtime.locate_file(entry))), entry.hash.value


def _binary_digests(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
    return digest.hexdigest(), encoded


def _trusted_binary(config: RLMConfig) -> str:
    if config.monty_binary_path is None:
        candidate, recorded_digest = _packaged_worker()
    else:
        candidate, recorded_digest = Path(config.monty_binary_path), None
        if not candidate.is_absolute():
            raise ValueError("monty_binary_path must be absolute")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("trusted Monty worker binary was not found") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise PermissionError("trusted Monty worker path is not an executable file")
    if recorded_digest is not None or config.monty_binary_sha256 is not None:
        hex_digest, encoded_digest = _binary_digests(resolved)
        if recorded_digest is not None and not hmac.compare_digest(encoded_digest, recorded_digest):
            raise PermissionError("Monty worker binary does not match its installed package manifest")
        if expected := config.monty_binary_sha256:
            if hmac.compare_digest(hex_digest, expected):
                return str(resolved)
            raise PermissionError("Monty worker binary SHA-256 does not match the configured allow-list")
    return str(resolved)


async def _acquire_sandbox_gate(timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if _SANDBOX_GATE.acquire(blocking=False):
            return True
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.01, remaining))


def _validate_code(code: str, config: RLMConfig) -> str:
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    normalized = textwrap.dedent(code).strip()
    size = len(normalized.encode("utf-8"))
    if size > config.max_code_bytes:
        raise ValueError(f"code exceeds max_code_bytes ({config.max_code_bytes})")
    return normalized


@dataclass(frozen=True)
class _CapturedOutput:
    stdout: str
    stderr: str
    emitted_bytes: int
    retained_bytes: int
    truncated: bool


def _utf8_prefix(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _split_output(collector: CollectStreams, config: RLMConfig) -> _CapturedOutput:
    stdout: list[str] = []
    stderr: list[str] = []
    remaining_bytes = config.max_output_bytes
    remaining_chars = config.truncate_output_chars
    emitted_bytes = 0
    retained_bytes = 0
    truncated = False
    for stream, value in collector.output:
        encoded = value.encode("utf-8")
        emitted_bytes += len(encoded)
        if remaining_bytes <= 0 or remaining_chars <= 0:
            truncated = truncated or bool(value)
            continue
        retained = value[:remaining_chars]
        retained = _utf8_prefix(retained, remaining_bytes)
        retained_size = len(retained.encode("utf-8"))
        retained_bytes += retained_size
        remaining_bytes -= retained_size
        remaining_chars -= len(retained)
        truncated = truncated or retained != value
        (stdout if stream == "stdout" else stderr).append(retained)
    return _CapturedOutput(
        stdout="".join(stdout),
        stderr="".join(stderr),
        emitted_bytes=emitted_bytes,
        retained_bytes=retained_bytes,
        truncated=truncated,
    )


def _bounded_text(value: str, config: RLMConfig) -> str:
    suffix = "\n... (output truncated)"
    max_chars = config.truncate_output_chars
    if len(value) > max_chars:
        suffix_chars = suffix[:max_chars]
        value = value[: max(0, max_chars - len(suffix_chars))] + suffix_chars
    encoded = value.encode("utf-8")
    if len(encoded) > config.max_output_bytes:
        encoded_suffix = suffix.encode("utf-8")[: config.max_output_bytes]
        prefix_size = max(0, config.max_output_bytes - len(encoded_suffix))
        value = encoded[:prefix_size].decode("utf-8", errors="ignore") + encoded_suffix.decode(
            "utf-8", errors="ignore"
        )
    return value


def _with_truncation_notice(value: str, captured: _CapturedOutput, config: RLMConfig) -> str:
    if not captured.truncated:
        return value
    notice = (
        f"\n... [printed output safely truncated: {captured.emitted_bytes} bytes emitted; "
        f"return capped at {config.max_output_bytes} bytes. REPL state is still available; "
        "use counts, aggregation, top-k results, IDs, or bounded excerpts next.]"
    )
    notice = _utf8_prefix(notice[: config.truncate_output_chars], config.max_output_bytes)
    content_char_budget = max(0, config.truncate_output_chars - len(notice))
    content_byte_budget = max(0, config.max_output_bytes - len(notice.encode("utf-8")))
    prefix = _utf8_prefix(value[:content_char_budget], content_byte_budget)
    return f"{prefix}{notice}"


def _append_result(stdout: str, value: Any, config: RLMConfig) -> str:
    if value is None:
        return _bounded_text(stdout, config)
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<{type(value).__name__}>"
    return _bounded_text(f"{stdout}{rendered}\n", config)


def _error_text(exc: BaseException) -> str:
    if isinstance(exc, MontyRuntimeError):
        return exc.display(format="type-msg")
    if isinstance(exc, MontyError):
        display = getattr(exc, "display", None)
        if callable(display):
            with contextlib.suppress(Exception):
                return str(display(format="type-msg"))
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def _is_terminal(exc: BaseException) -> bool:
    if isinstance(exc, MontyCrashedError):
        return True
    if isinstance(exc, MontyRuntimeError):
        with contextlib.suppress(Exception):
            return isinstance(exc.exception(), (MemoryError, TimeoutError))
    return False


def _failure_kind(exc: BaseException, error: str, config: RLMConfig) -> str:
    if isinstance(exc, MontyCrashedError):
        return "worker_crash"
    if isinstance(exc, MontyRuntimeError):
        with contextlib.suppress(Exception):
            runtime_exception = exc.exception()
            if isinstance(runtime_exception, MemoryError):
                hard_limit = f"> {config.max_emitted_output_bytes} bytes"
                return "output_flood" if hard_limit in error else "memory_limit"
            if isinstance(runtime_exception, TimeoutError):
                return "execution_timeout"
    return "code_error"


async def _close_async_context(
    context_manager: Any | None,
) -> tuple[asyncio.CancelledError | None, Exception | None]:
    """Close a resource while retaining cancellation until all cleanup runs."""
    if context_manager is None:
        return None, None
    try:
        await context_manager.__aexit__(None, None, None)
    except asyncio.CancelledError as exc:
        return exc, None
    except Exception as exc:
        return None, exc
    return None, None


class REPLEnvironment:
    """Synchronous, stateful Monty sandbox.

    A dedicated worker process is used for the lifetime of this object and is
    recycled after the session closes. No filesystem, environment, network,
    host object, or OS capability is exposed to sandboxed code.
    """

    def __init__(self, context: ContextType, config: RLMConfig | None = None):
        self.config = config or RLMConfig()
        self._context: ContextType | None = self.config.context_for_monty(context)
        self._context_pending = True
        self._submodel_calls = 0
        self._closed = False
        self._poisoned = False
        self._pool_cm: Monty | None = None
        self._pool: Monty | None = None
        self._session_cm: MontySession | None = None
        self._session: MontySession | None = None
        self._gate_acquired = False
        self._open()

    def _open(self) -> None:
        binary_path = _trusted_binary(self.config)
        if not _SANDBOX_GATE.acquire(timeout=self.config.checkout_timeout):
            raise TimeoutError("timed out waiting for the process-wide sandbox capacity limit")
        self._gate_acquired = True
        pool_cm: Monty | None = None
        try:
            pool_cm = Monty(
                binary_path=binary_path,
                min_processes=1,
                max_processes=1,
                checkout_timeout=self.config.checkout_timeout,
                request_timeout=self.config.code_timeout,
                max_checkouts_per_worker=1,
            )
            pool = pool_cm.__enter__()
            session_cm = pool.checkout(limits=_limits(self.config))
            session = session_cm.__enter__()
        except BaseException:
            if pool_cm is not None:
                with contextlib.suppress(Exception):
                    pool_cm.__exit__(None, None, None)
            if self._gate_acquired:
                _SANDBOX_GATE.release()
                self._gate_acquired = False
            raise
        self._pool_cm = pool_cm
        self._pool = pool
        self._session_cm = session_cm
        self._session = session

    def _llm_query(self, prompt: str) -> str:
        from .logging import get_logger

        if not isinstance(prompt, str):
            raise TypeError("llm_query prompt must be a string")
        if not self.config.sub_model:
            raise RuntimeError("No sub-model configured")
        if self._submodel_calls >= self.config.max_submodel_calls:
            raise RuntimeError("llm_query call limit exceeded")
        if len(prompt.encode("utf-8")) > self.config.max_submodel_prompt_bytes:
            raise ValueError("llm_query prompt exceeds configured limit")
        self._submodel_calls += 1
        logger = get_logger()
        logger.log_llm_query(prompt)
        try:
            response = model_request_sync(
                self.config.sub_model,
                [ModelRequest.user_text_prompt(prompt)],
                model_settings=_submodel_settings(self.config),
            )
        except Exception:
            raise RuntimeError("sub-model request failed") from None
        result = "".join(part.content for part in response.parts if isinstance(part, TextPart))
        if len(result.encode("utf-8")) > self.config.max_submodel_output_bytes:
            raise ValueError("llm_query response exceeds configured limit")
        logger.log_llm_response(result)
        return result

    def execute(self, code: str) -> REPLResult:
        if self._closed or self._poisoned or self._session is None:
            raise RuntimeError("sandbox session is closed or unusable")
        normalized = _validate_code(code, self.config)
        collector = CollectStreams(max_bytes=self.config.max_emitted_output_bytes)
        inputs = {"context": self._context} if self._context_pending else None
        # The worker owns the only remaining copy after this feed. Keeping a
        # host-side reference would double peak tenant-data retention.
        self._context = None
        external_lookup = {"llm_query": self._llm_query} if self.config.sub_model else None
        started = time.perf_counter()
        try:
            self._context_pending = False
            value = self._session.feed_run(
                normalized,
                inputs=inputs,
                external_lookup=external_lookup,
                print_callback=collector,
            )
            captured = _split_output(collector, self.config)
            stdout = _append_result(captured.stdout, value, self.config)
            stdout = _with_truncation_notice(stdout, captured, self.config)
            return REPLResult(
                stdout=stdout,
                stderr=_bounded_text(captured.stderr, self.config),
                locals={},
                execution_time=time.perf_counter() - started,
                output_truncated=captured.truncated,
                emitted_output_bytes=captured.emitted_bytes,
            )
        except MontyError as exc:
            captured = _split_output(collector, self.config)
            terminal = _is_terminal(exc)
            if terminal:
                self._poisoned = True
                self.cleanup()
            error = _error_text(exc)
            failure_kind = _failure_kind(exc, error, self.config)
            stdout = _with_truncation_notice(_bounded_text(captured.stdout, self.config), captured, self.config)
            return REPLResult(
                stdout=stdout,
                stderr=_bounded_text(f"{captured.stderr}\nError: {error}".lstrip(), self.config),
                locals={},
                execution_time=time.perf_counter() - started,
                success=False,
                output_truncated=captured.truncated,
                emitted_output_bytes=captured.emitted_bytes,
                fatal=terminal,
                failure_kind=failure_kind,
            )

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context = None
        if self._session_cm is not None:
            with contextlib.suppress(Exception):
                self._session_cm.__exit__(None, None, None)
        if self._pool_cm is not None:
            with contextlib.suppress(Exception):
                self._pool_cm.__exit__(None, None, None)
        self._session = None
        self._session_cm = None
        self._pool = None
        self._pool_cm = None
        if self._gate_acquired:
            _SANDBOX_GATE.release()
            self._gate_acquired = False

    close = cleanup

    def __enter__(self) -> REPLEnvironment:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.cleanup()


class AsyncREPLEnvironment:
    """Async per-run Monty sandbox used by the Pydantic AI toolset."""

    def __init__(self, context: ContextType, config: RLMConfig):
        self.config = config
        self._context: ContextType | None = config.context_for_monty(context)
        self._context_pending = True
        self._submodel_calls = 0
        self._closed = False
        self._poisoned = False
        self._pool_cm: AsyncMonty | None = None
        self._pool: AsyncMonty | None = None
        self._session_cm: AsyncMontySession | None = None
        self._session: AsyncMontySession | None = None
        self._gate_acquired = False

    async def open(self) -> AsyncREPLEnvironment:
        if self._pool is not None:
            return self
        binary_path = _trusted_binary(self.config)
        if not await _acquire_sandbox_gate(self.config.checkout_timeout):
            raise TimeoutError("timed out waiting for the process-wide sandbox capacity limit")
        self._gate_acquired = True
        pool_cm: AsyncMonty | None = None
        try:
            pool_cm = AsyncMonty(
                binary_path=binary_path,
                min_processes=1,
                max_processes=1,
                checkout_timeout=self.config.checkout_timeout,
                request_timeout=self.config.code_timeout,
                max_checkouts_per_worker=1,
            )
            pool = await pool_cm.__aenter__()
            session_cm = pool.checkout(limits=_limits(self.config))
            session = await session_cm.__aenter__()
        except BaseException:
            # Retain the original startup failure/cancellation, but never let a
            # second cancellation interrupt worker and gate cleanup.
            self._closed = True
            self._context = None
            await _close_async_context(pool_cm)
            if self._gate_acquired:
                _SANDBOX_GATE.release()
                self._gate_acquired = False
            raise
        self._pool_cm = pool_cm
        self._pool = pool
        self._session_cm = session_cm
        self._session = session
        return self

    async def _llm_query(self, prompt: str) -> str:
        from .logging import get_logger

        if not isinstance(prompt, str):
            raise TypeError("llm_query prompt must be a string")
        if not self.config.sub_model:
            raise RuntimeError("No sub-model configured")
        if self._submodel_calls >= self.config.max_submodel_calls:
            raise RuntimeError("llm_query call limit exceeded")
        if len(prompt.encode("utf-8")) > self.config.max_submodel_prompt_bytes:
            raise ValueError("llm_query prompt exceeds configured limit")
        self._submodel_calls += 1
        logger = get_logger()
        logger.log_llm_query(prompt)
        try:
            response = await model_request(
                self.config.sub_model,
                [ModelRequest.user_text_prompt(prompt)],
                model_settings=_submodel_settings(self.config),
            )
        except Exception:
            raise RuntimeError("sub-model request failed") from None
        result = "".join(part.content for part in response.parts if isinstance(part, TextPart))
        if len(result.encode("utf-8")) > self.config.max_submodel_output_bytes:
            raise ValueError("llm_query response exceeds configured limit")
        logger.log_llm_response(result)
        return result

    async def execute(self, code: str) -> REPLResult:
        if self._closed or self._poisoned or self._session is None:
            raise RuntimeError("sandbox session is closed or unusable")
        normalized = _validate_code(code, self.config)
        collector = CollectStreams(max_bytes=self.config.max_emitted_output_bytes)
        inputs = {"context": self._context} if self._context_pending else None
        # Drop host-side tenant data before crossing the async cancellation
        # boundary. The local ``inputs`` reference dies when this call returns.
        self._context = None
        external_lookup = {"llm_query": self._llm_query} if self.config.sub_model else None
        started = time.perf_counter()
        try:
            self._context_pending = False
            value = await self._session.feed_run(
                normalized,
                inputs=inputs,
                external_lookup=external_lookup,
                print_callback=collector,
            )
            captured = _split_output(collector, self.config)
            stdout = _append_result(captured.stdout, value, self.config)
            stdout = _with_truncation_notice(stdout, captured, self.config)
            return REPLResult(
                stdout=stdout,
                stderr=_bounded_text(captured.stderr, self.config),
                locals={},
                execution_time=time.perf_counter() - started,
                output_truncated=captured.truncated,
                emitted_output_bytes=captured.emitted_bytes,
            )
        except MontyError as exc:
            captured = _split_output(collector, self.config)
            terminal = _is_terminal(exc)
            if terminal:
                self._poisoned = True
                await self.close()
            error = _error_text(exc)
            failure_kind = _failure_kind(exc, error, self.config)
            stdout = _with_truncation_notice(_bounded_text(captured.stdout, self.config), captured, self.config)
            return REPLResult(
                stdout=stdout,
                stderr=_bounded_text(f"{captured.stderr}\nError: {error}".lstrip(), self.config),
                locals={},
                execution_time=time.perf_counter() - started,
                success=False,
                output_truncated=captured.truncated,
                emitted_output_bytes=captured.emitted_bytes,
                fatal=terminal,
                failure_kind=failure_kind,
            )

    async def close(self) -> None:
        if self._closed and self._session_cm is None and self._pool_cm is None:
            return
        self._closed = True
        self._context = None
        session_cancellation, session_error = await _close_async_context(self._session_cm)
        pool_cancellation, pool_error = await _close_async_context(self._pool_cm)
        cancellation = session_cancellation or pool_cancellation
        cleanup_error = session_error or pool_error
        self._session = None
        self._session_cm = None
        self._pool = None
        self._pool_cm = None
        if self._gate_acquired:
            _SANDBOX_GATE.release()
            self._gate_acquired = False
        if cancellation is not None:
            raise cancellation
        if cleanup_error is not None:
            raise RuntimeError("failed to fully close Monty sandbox resources") from cleanup_error

    async def __aenter__(self) -> AsyncREPLEnvironment:
        return await self.open()

    async def __aexit__(self, *args: object) -> None:
        await self.close()
