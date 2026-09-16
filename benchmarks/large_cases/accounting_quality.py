"""Shared, question-specific truth validation for both accounting runners."""

from __future__ import annotations

import re

from .accounting_data import DATASET_SIZE, cutoff_invoice_id


def _contains_all(text: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    return all(any(term in text for term in alternatives) for alternatives in groups)


def accounting_domain_quality_error(answer: str, *, record_count: int) -> str | None:
    """Return a precise diagnostic when any of the five accounting answers is wrong or absent."""
    if record_count != DATASET_SIZE:
        return f"accounting dataset has {record_count}/{DATASET_SIZE} records"
    text = answer.lower().replace(",", ".").replace("−", "-").replace("–", "-").replace("\\%", "%")  # noqa: RUF001
    text = text.replace("*", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    # Accept equivalent presentation precision while retaining canonical truth
    # values for the question-specific checks below.
    text = re.sub(r"\b40\.0+\s*%", "40%", text)
    text = re.sub(r"\b35\.2\d*\s*%", "35.2%", text)
    text = re.sub(r"-?4\.7[89]\d*\s+(?:percentage points?|pp|procentpoint)", "4.8 percentage point", text)
    text = re.sub(r"\b15\.7\d*\s*%", "15.7%", text)
    text = re.sub(r"\b64\.8\d*\b", "64.9", text)
    text = re.sub(r"\b103\.4\d*\b", "103.4", text)
    text = re.sub(r"\b38\.5\d*\b", "38.5", text)
    question_checks = {
        "Q1": (
            ("40.0%", "40%"),
            ("35.2%",),
            ("4.8 percentage point", "4.8 pp", "4.8 procentpoint"),
            ("hardware",),
            ("component", "supplier"),
            ("freight",),
        ),
        "Q2": (
            ("18.1",),
            ("21.7",),
            ("15.7%",),
            ("2.4",),
            ("1.2",),
            ("restructur",),
            ("legal", "settlement", "dispute"),
        ),
        "Q3": (
            ("64.9",),
            ("103.4",),
            ("38.5",),
            ("receivable", "dso"),
            ("inventory", "dio"),
        ),
        "Q4": (
            ("1.9",),
            ("-8.6",),
            ("26.4",),
            ("0.95",),
            ("44.5",),
            ("free cash flow", "fcf"),
            ("liquidity",),
        ),
        "Q5": (
            ("4.8",),
            ("133.2",),
            ("7.4",),
            ("11.3",),
            ("debit revenue", "debit software revenue", "dr revenue", "dr software revenue"),
            ("credit deferred revenue", "cr deferred revenue"),
        ),
    }
    for question, groups in question_checks.items():
        if not _contains_all(text, groups):
            return f"{question} is missing one or more required calculations or conclusions"
    twelve_exceptions = re.search(r"\b12(?:\s+[a-z-]+){0,3}\s+(?:contracts|invoices|entries|exceptions)\b", text) or re.search(
        r"\b(?:number|count)\s+of\s+(?:contracts|invoices|entries|exceptions)\s*[:=]\s*12\b", text
    )
    twelve_exceptions = twelve_exceptions or re.search(r"\b(?:contracts|invoices|entries|exceptions)\s*[:=]\s*12\b", text)
    if twelve_exceptions is None:
        return "Q5 does not identify exactly 12 cutoff exceptions"
    valid_ids = {cutoff_invoice_id(index).lower() for index in range(1, 13)}
    cited_ids = {value.lower() for value in re.findall(r"INV-CUT-2025-\d{3}", answer, flags=re.IGNORECASE)}
    if len(valid_ids & cited_ids) < 3:
        return "Q5 cites fewer than three valid affected invoice IDs"
    return None
