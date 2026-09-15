from __future__ import annotations

from pydantic_ai_rlm import arithmetic_consistency_errors


def test_arithmetic_validator_detects_accounting_failure() -> None:
    answer = r"Reported EBITDA = 14.6 + 3.5 = 17.1\text{m}. Normalized EBITDA = 17.1 + 2.4 + 1.2 = 20.7."

    errors = arithmetic_consistency_errors(answer)

    assert any("14.6 + 3.5=17.1" in error for error in errors)


def test_arithmetic_validator_accepts_rounding_percentages_and_negative_values() -> None:
    answer = (
        r"48.6 / 138.0 = 35.22\%. "
        r"35.217 - 40.000 = -4.783\text{ percentage points}. "
        r"1.9 + (-10.5) = -8.6. "
        r"20.6 / 21.7 = 0.95."
    )

    assert arithmetic_consistency_errors(answer) == []


def test_arithmetic_validator_does_not_execute_arbitrary_text() -> None:
    answer = "__import__('os').system('whoami') = 0 and invoice INV-CUT-2025-001"

    assert arithmetic_consistency_errors(answer) == []
