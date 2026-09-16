"""Deterministic synthetic annual-report and accounting-ledger corpus."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

RANDOM_SEED = 20260913
ROUTINE_LEDGER_ENTRIES = 3_600
DATASET_PATH = Path(__file__).resolve().parent / "data" / "synthetic_accounting.jsonl"
EXPECTED_SHA256 = "92a537609a9fb97251a98040c00479dce09bab9453abf6037c061fc671756d58"

TASK = """Analyze the synthetic 2025 annual-report and audit-register records for NordicFlow A/S. All statement
amounts are DKK millions and reported figures should be used before the proposed cutoff adjustment unless a question
says otherwise. Answer all five questions separately, show each formula, and cite the record IDs used.

Q1. Calculate reported consolidated gross margin for 2024 and 2025, the percentage-point movement, and identify the
segment that economically caused the deterioration.
Q2. Calculate 2025 reported EBITDA, normalized EBITDA and normalized EBITDA margin. Reconcile every adjustment and
do not treat recurring expenses as one-offs.
Q3. Using the company's stated year-end-balance convention and 365 days, calculate DSO, DIO, DPO and the cash
conversion cycle for 2024 and 2025. Quantify the change and identify the principal working-capital drivers.
Q4. Reconcile 2025 cash flow from operations from the indirect bridge, calculate free cash flow, year-end total
liquidity, net-debt-to-normalized-EBITDA, and additional net-debt headroom under the 3.0x covenant.
Q5. Test the December software revenue entries against the stated revenue-recognition policy. Identify the exact
number and total value of exceptions, quantify corrected 2025 revenue, pre-tax profit and deferred revenue, and give
the correcting journal entry. Cite at least three affected invoice IDs."""

GROUND_TRUTH = {
    "gross_margin_2024_pct": 40.0,
    "gross_margin_2025_pct": 35.2,
    "gross_margin_change_pp": -4.8,
    "reported_ebitda_m": 18.1,
    "normalized_ebitda_m": 21.7,
    "normalized_ebitda_margin_pct": 15.7,
    "ccc_2024_days": 64.9,
    "ccc_2025_days": 103.4,
    "ccc_change_days": 38.5,
    "cfo_m": 1.9,
    "free_cash_flow_m": -8.6,
    "liquidity_m": 26.4,
    "net_leverage": 0.95,
    "covenant_headroom_m": 44.5,
    "cutoff_count": 12,
    "cutoff_total_m": 4.8,
    "corrected_revenue_m": 133.2,
    "corrected_pretax_profit_m": 7.4,
    "corrected_deferred_revenue_m": 11.3,
}


def cutoff_invoice_id(index: int) -> str:
    """Return a stable identifier for one intentionally misperiodized contract."""
    return f"INV-CUT-2025-{index:03d}"


def _statement_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "record_id": "META-COMPANY-001",
            "record_type": "company_profile",
            "entity": "NordicFlow A/S",
            "fiscal_year_end": "2025-12-31",
            "currency": "DKK millions for statement and note records; DKK for transaction records",
            "status": "entirely synthetic benchmark data",
        },
        {
            "record_id": "POL-WC-001",
            "record_type": "accounting_policy",
            "topic": "working_capital_metrics",
            "text": (
                "Management calculates DSO, DIO and DPO from fiscal year-end balances, not averages, using 365 "
                "days. DSO uses revenue; DIO and DPO use cost of goods sold. CCC = DSO + DIO - DPO."
            ),
        },
        {
            "record_id": "POL-REV-001",
            "record_type": "accounting_policy",
            "topic": "software_subscription_revenue",
            "text": (
                "Annual software subscriptions are recognized straight-line from the service-start date through the "
                "service-end date. Amounts invoiced before service starts remain deferred revenue; invoice date and "
                "cash collection do not accelerate recognition."
            ),
        },
    ]
    income_statement = {
        "2024": {
            "hardware_revenue": 80.0,
            "software_revenue": 40.0,
            "total_revenue": 120.0,
            "hardware_cogs": 56.0,
            "software_cogs": 16.0,
            "total_cogs": 72.0,
            "gross_profit": 48.0,
            "sales_and_marketing": 10.5,
            "research_and_development": 8.0,
            "general_and_administrative": 7.0,
            "depreciation_and_amortization": 3.0,
            "operating_profit": 19.5,
            "interest_expense": 2.0,
            "profit_before_tax": 17.5,
            "tax_expense": 4.4,
            "net_income": 13.1,
        },
        "2025": {
            "hardware_revenue": 90.0,
            "software_revenue": 48.0,
            "total_revenue": 138.0,
            "hardware_cogs": 70.2,
            "software_cogs": 19.2,
            "total_cogs": 89.4,
            "gross_profit": 48.6,
            "sales_and_marketing": 12.0,
            "research_and_development": 9.5,
            "general_and_administrative": 9.0,
            "depreciation_and_amortization": 3.5,
            "operating_profit": 14.6,
            "interest_expense": 2.4,
            "profit_before_tax": 12.2,
            "tax_expense": 3.05,
            "net_income": 9.15,
        },
    }
    for period, values in income_statement.items():
        for line_item, amount in values.items():
            records.append(
                {
                    "record_id": f"STAT-IS-{period}-{line_item.upper()}",
                    "record_type": "financial_statement_line",
                    "statement": "income_statement",
                    "period": period,
                    "line_item": line_item,
                    "amount_m_dkk": amount,
                    "basis": "reported_before_proposed_audit_adjustments",
                }
            )
    balance_sheet = {
        "2024": {
            "cash": 16.0,
            "trade_receivables": 18.0,
            "inventory": 14.0,
            "trade_payables": 12.0,
            "deferred_revenue": 4.0,
            "interest_bearing_debt": 27.0,
            "other_operating_accruals": 5.0,
        },
        "2025": {
            "cash": 11.4,
            "trade_receivables": 27.6,
            "inventory": 22.35,
            "trade_payables": 14.9,
            "deferred_revenue": 6.5,
            "interest_bearing_debt": 32.0,
            "other_operating_accruals": 6.0,
        },
    }
    for period, values in balance_sheet.items():
        for line_item, amount in values.items():
            records.append(
                {
                    "record_id": f"STAT-BS-{period}-{line_item.upper()}",
                    "record_type": "financial_statement_line",
                    "statement": "balance_sheet",
                    "period": period,
                    "line_item": line_item,
                    "amount_m_dkk": amount,
                    "basis": "reported_before_proposed_audit_adjustments",
                }
            )
    cash_flow = {
        "net_income": 9.15,
        "depreciation_and_amortization": 3.5,
        "share_based_compensation": 0.8,
        "increase_in_trade_receivables": -9.6,
        "increase_in_inventory": -8.35,
        "increase_in_trade_payables": 2.9,
        "increase_in_deferred_revenue": 2.5,
        "increase_in_other_operating_accruals": 1.0,
        "cash_flow_from_operations": 1.9,
        "capital_expenditure": -10.5,
        "free_cash_flow": -8.6,
        "new_debt_drawn": 5.0,
        "dividends_paid": -1.0,
        "net_change_in_cash": -4.6,
        "closing_cash": 11.4,
    }
    for line_item, amount in cash_flow.items():
        records.append(
            {
                "record_id": f"STAT-CF-2025-{line_item.upper()}",
                "record_type": "financial_statement_line",
                "statement": "cash_flow_statement",
                "period": "2025",
                "line_item": line_item,
                "amount_m_dkk": amount,
            }
        )
    records.extend(
        [
            {
                "record_id": "NOTE-SEG-HARDWARE-2025",
                "record_type": "segment_note",
                "segment": "hardware",
                "period": "2025",
                "revenue_m_dkk": 90.0,
                "cogs_m_dkk": 70.2,
                "comment": (
                    "Component purchase prices and inbound freight rose faster than selling prices; there was no "
                    "equivalent deterioration in software delivery economics."
                ),
            },
            {
                "record_id": "NOTE-SEG-SOFTWARE-2025",
                "record_type": "segment_note",
                "segment": "software",
                "period": "2025",
                "revenue_m_dkk": 48.0,
                "cogs_m_dkk": 19.2,
                "comment": "Subscription gross margin remained stable year over year.",
            },
            {
                "record_id": "NOTE-ADJ-RESTRUCTURE-2025",
                "record_type": "ebitda_adjustment_note",
                "amount_m_dkk": 2.4,
                "income_statement_location": "general_and_administrative",
                "treatment": "non-recurring restructuring charge eligible for normalization",
            },
            {
                "record_id": "NOTE-ADJ-LEGAL-2025",
                "record_type": "ebitda_adjustment_note",
                "amount_m_dkk": 1.2,
                "income_statement_location": "general_and_administrative",
                "treatment": "one-time settlement of a legacy dispute eligible for normalization",
            },
            {
                "record_id": "NOTE-OPEX-RECURRING-2025",
                "record_type": "ebitda_adjustment_note",
                "amount_m_dkk": 0.8,
                "income_statement_location": "research_and_development",
                "treatment": "recurring cloud migration expense; explicitly not eligible for normalization",
            },
            {
                "record_id": "NOTE-DEBT-2025",
                "record_type": "debt_and_liquidity_note",
                "cash_m_dkk": 11.4,
                "interest_bearing_debt_m_dkk": 32.0,
                "undrawn_committed_revolver_m_dkk": 15.0,
                "net_debt_covenant_limit_x": 3.0,
                "covenant_ebitda_basis": "normalized EBITDA",
            },
        ]
    )
    return records


_ROUTINE_TEMPLATES = (
    ("6200", "freight_expense", "2000", "trade_payables", "Inbound freight settlement"),
    ("1100", "trade_receivables", "4000", "hardware_revenue", "Hardware customer invoice"),
    ("1100", "trade_receivables", "4100", "software_revenue", "Monthly software service invoice"),
    ("5100", "component_cost", "2000", "trade_payables", "Component supplier invoice"),
    ("1200", "inventory", "2000", "trade_payables", "Finished goods receipt"),
    ("7100", "research_and_development", "1000", "cash", "Recurring engineering service"),
    ("7200", "sales_and_marketing", "2000", "trade_payables", "Campaign and channel expense"),
    ("7300", "general_and_administrative", "1000", "cash", "Ordinary administrative expense"),
)


def _routine_ledger_records() -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    start = date(2025, 1, 1)
    records: list[dict[str, Any]] = []
    for index in range(1, ROUTINE_LEDGER_ENTRIES + 1):
        debit_code, debit_name, credit_code, credit_name, description = rng.choice(_ROUTINE_TEMPLATES)
        posting_date = start + timedelta(days=(index * 37) % 365)
        amount = rng.randrange(25_000, 475_001, 500)
        is_subscription = credit_name == "software_revenue"
        records.append(
            {
                "record_id": f"JE-2025-{index:05d}",
                "record_type": "journal_entry",
                "posting_date": posting_date.isoformat(),
                "debit_account": f"{debit_code} {debit_name}",
                "credit_account": f"{credit_code} {credit_name}",
                "amount_dkk": amount,
                "description": description,
                "counterparty_id": f"CP-{(index * 17) % 613:04d}",
                "invoice_id": f"INV-2025-{index:05d}",
                "service_start": posting_date.isoformat() if is_subscription else None,
                "service_end": ((posting_date + timedelta(days=29)).isoformat() if is_subscription else None),
                "register_scope": "supporting transaction register; statement records contain authoritative totals",
            }
        )
    return records


def _cutoff_records() -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"JE-CUT-2025-{index:03d}",
            "record_type": "journal_entry",
            "posting_date": f"2025-12-{18 + index:02d}",
            "debit_account": "1100 trade_receivables",
            "credit_account": "4100 software_revenue",
            "amount_dkk": 400_000,
            "description": "Annual enterprise subscription invoiced and recognized in full on posting",
            "counterparty_id": f"ENTERPRISE-{index:03d}",
            "invoice_id": cutoff_invoice_id(index),
            "service_start": "2026-01-01",
            "service_end": "2026-12-31",
            "revenue_recognized_m_dkk_2025": 0.4,
            "register_scope": "supporting transaction register; statement records contain authoritative totals",
        }
        for index in range(1, 13)
    ]


def records() -> list[dict[str, Any]]:
    """Return the complete deterministic corpus with exceptions scattered among routine entries."""
    ledger = _routine_ledger_records() + _cutoff_records()
    random.Random(RANDOM_SEED + 1).shuffle(ledger)
    return _statement_records() + ledger


DATASET_SIZE = len(records())


def ensure_dataset(path: Path = DATASET_PATH) -> Path:
    """Write the deterministic JSONL corpus only when it is absent or stale."""
    expected = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records())
    if not path.exists() or path.read_text(encoding="utf-8") != expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="")
    return path


def load_dataset(path: Path = DATASET_PATH) -> str:
    """Ensure and return the accounting JSONL supplied to either benchmark."""
    content = ensure_dataset(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"accounting fixture digest mismatch: {digest}")
    return content
