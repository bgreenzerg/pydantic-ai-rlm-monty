"""Shared answer-truth validation for feedback benchmark variants."""

from __future__ import annotations

import re

from .feedback_data import COHORT_INDICES, DATASET_SIZE, SUPPORTING_COUNT, feedback_id


def feedback_domain_quality_error(answer: str, *, record_count: int) -> str | None:
    """Return the shared factual-answer failure diagnostic, if any."""
    if record_count != DATASET_SIZE:
        return f"feedback dataset has {record_count}/{DATASET_SIZE} records"
    answer_lower = answer.lower()
    required_groups = (
        ("checkout",),
        ("address",),
        ("autocomplete", "suggestion"),
        ("apartment", "unit", "door"),
        ("danish", "denmark", "da-dk"),
        ("mobile safari", "mobile_safari"),
        ("preserve", "retain", "keep"),
    )
    if any(not any(term in answer_lower for term in group) for group in required_groups):
        return "answer is missing root cause, affected pattern, count, or remediation"
    explicit_count = re.search(
        rf"\b{SUPPORTING_COUNT}\s+(?:supporting\s+)?(?:reports|comments|records|cases)\b",
        answer_lower,
    ) or re.search(rf"\b{SUPPORTING_COUNT}\s+(?:of|/)\s+120\b", answer_lower)
    explicit_count = explicit_count or re.search(
        rf"\b(?:supporting\s+)?(?:count|total)\s*(?:is|=|:)?\s*{SUPPORTING_COUNT}\b",
        answer_lower,
    )
    if explicit_count is None:
        return f"answer does not state an explicit supporting count of {SUPPORTING_COUNT}"
    cohort_ids = {feedback_id(index).lower() for index in COHORT_INDICES}
    cited_ids = {value.lower() for value in re.findall(r"FB-2026-\d{4}", answer, flags=re.IGNORECASE)}
    valid_cited_ids = cohort_ids & cited_ids
    if len(valid_cited_ids) < 3:
        return f"answer cites only {len(valid_cited_ids)} valid cohort feedback IDs (expected at least 3)"
    return None
