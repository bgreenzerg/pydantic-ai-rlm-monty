from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic_ai_rlm import RLMConfig, configure_logging


def test_upstream_prompts_are_byte_identical() -> None:
    prompts = Path(__file__).parents[1] / "src" / "pydantic_ai_rlm" / "prompts.py"
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    assert digest == "3be7a348f185658339c68133192ad4d35331b271eaaae4a6e8737bcdc10b8885"


def test_logging_redacts_content_by_default(capsys) -> None:
    logger = configure_logging(enabled=True)
    logger.log_code_execution("bank-central-secret")
    logger.log_llm_query("customer-secret")
    output = capsys.readouterr().out

    assert "bank-central-secret" not in output
    assert "customer-secret" not in output
    assert "content redacted" in output


def test_logging_reports_truncation_without_content(capsys) -> None:
    from pydantic_ai_rlm import REPLResult

    logger = configure_logging(enabled=True)
    logger.log_result(
        REPLResult(
            stdout="bank-central-secret",
            stderr="",
            locals={},
            execution_time=0.1,
            output_truncated=True,
            emitted_output_bytes=100_000,
        )
    )
    output = capsys.readouterr().out

    assert "bank-central-secret" not in output
    assert "emitted_output_bytes=100000" in output
    assert "output_truncated=True" in output


def test_defaults_are_all_bounded() -> None:
    config = RLMConfig()
    assert config.code_timeout > 0
    assert config.max_context_bytes > 0
    assert config.max_code_bytes > 0
    assert config.max_output_bytes > 0
    assert config.max_emitted_output_bytes >= config.max_output_bytes
    assert config.max_memory_bytes > 0
    assert config.max_total_execution_seconds > 0
    assert config.max_suspensions > 0
