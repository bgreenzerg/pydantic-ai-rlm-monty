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


def test_defaults_are_all_bounded() -> None:
    config = RLMConfig()
    assert config.code_timeout > 0
    assert config.max_context_bytes > 0
    assert config.max_code_bytes > 0
    assert config.max_output_bytes > 0
    assert config.max_memory_bytes > 0
    assert config.max_total_execution_seconds > 0
    assert config.max_suspensions > 0
