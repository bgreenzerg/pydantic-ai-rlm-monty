from __future__ import annotations

import time

from pydantic_ai_rlm import arithmetic_consistency_errors, grounding_consistency_errors


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


def test_arithmetic_validator_preserves_decimal_literal_precision() -> None:
    answer = "0.123456789012345678 + 0.000000000000000001 = 0.123456789012345679"

    assert arithmetic_consistency_errors(answer) == []


def test_arithmetic_validator_accepts_ratio_and_whole_number_percent_notation() -> None:
    assert arithmetic_consistency_errors("48.6 / 138 = 35.22% and 25 * 4 = 100%") == []


def test_arithmetic_validator_is_bounded_on_adversarial_text() -> None:
    text = ("1 + " * 250_000) + "x"
    started = time.perf_counter()

    assert arithmetic_consistency_errors(text) == []
    assert time.perf_counter() - started < 1.0


def test_grounding_validator_accepts_exact_consecutive_quotes() -> None:
    context = {"report": "Revenue increased by 45 percent because demand expanded in Denmark."}
    info = "Revenue increased materially [1]."
    grounding = {"1": "Revenue increased by 45 percent"}

    assert grounding_consistency_errors(info, grounding, context) == []


def test_grounding_validator_rejects_fabricated_or_mismatched_citations() -> None:
    errors = grounding_consistency_errors(
        "Revenue increased [1] and margins improved [3].",
        {"1": "fabricated quote long enough", "2": "another fabricated quote"},
        "Revenue was flat in the source document.",
    )

    assert "citation markers and grounding keys do not match" in errors
    assert "citation keys must be consecutive starting at 1" not in errors
    assert any("not an exact quote" in error for error in errors)
