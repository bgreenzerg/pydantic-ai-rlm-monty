from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "benchmarks"))

from large_cases import accounting_data, feedback_data  # noqa: E402
from large_cases.accounting_quality import accounting_domain_quality_error  # noqa: E402
from large_cases.feedback_quality import feedback_domain_quality_error  # noqa: E402


def _render(records: list[dict[str, object]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)


def test_feedback_fixture_is_reproducible() -> None:
    content = _render(feedback_data.records())
    assert content.count("\n") == feedback_data.DATASET_SIZE == 5_000
    assert len(content.encode("utf-8")) == 1_557_639
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == feedback_data.EXPECTED_SHA256


def test_accounting_fixture_is_reproducible() -> None:
    content = _render(accounting_data.records())
    assert content.count("\n") == accounting_data.DATASET_SIZE == 3_682
    assert len(content.encode("utf-8")) == 1_654_972
    assert hashlib.sha256(content.encode("utf-8")).hexdigest() == accounting_data.EXPECTED_SHA256


def test_feedback_quality_gate_accepts_canonical_answer() -> None:
    identifiers = ", ".join(feedback_data.REPRESENTATIVE_IDS)
    answer = (
        "The checkout address autocomplete suggestion drops apartment, unit, or door data for Danish da-DK "
        f"mobile Safari users. The supporting count is 24 reports, including {identifiers}. "
        "Preserve the complete address and add a regression test."
    )
    assert feedback_domain_quality_error(answer, record_count=feedback_data.DATASET_SIZE) is None


def test_accounting_quality_gate_accepts_canonical_answer() -> None:
    answer = """
Q1: Gross margin was 40% and 35.2%, down 4.8 percentage points due to hardware component and freight costs.
Q2: EBITDA was 18.1; add restructuring 2.4 and legal settlement 1.2 for normalized EBITDA 21.7 and margin 15.7%.
Q3: CCC moved from 64.9 to 103.4 days, up 38.5, driven by receivables/DSO and inventory/DIO.
Q4: CFO was 1.9, free cash flow was -8.6, liquidity 26.4, leverage 0.95, and headroom 44.5.
Q5: 12 cutoff exceptions total 4.8; corrected revenue 133.2, pretax profit 7.4, deferred revenue 11.3.
Debit software revenue and credit deferred revenue. IDs: INV-CUT-2025-001, INV-CUT-2025-002, INV-CUT-2025-003.
"""
    assert accounting_domain_quality_error(answer, record_count=accounting_data.DATASET_SIZE) is None
