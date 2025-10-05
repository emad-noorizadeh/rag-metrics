# tests/test_entities.py
import math
import re
from typing import Dict, List, Tuple

import pytest

# Project imports
import extractor
from extractor import (
    Entity,
    MoneyValue,
    NumberValue,
    PercentValue,
    DateValue,
    QuantityValue,
    ExtractorConfig,
    DEFAULT_CONFIG,
)
from metric_utils import context_utilization_report_with_entities

TEST_CONFIG = ExtractorConfig(use_spacy_fusion=True)


# -------------------------
# Small helpers for asserts
# -------------------------
def _ents(text: str, config: ExtractorConfig = TEST_CONFIG) -> List[Entity]:
    return extractor.extract_entities(text, config=config)

def ents_by_type(text: str, config: ExtractorConfig = TEST_CONFIG) -> Dict[str, List[Entity]]:
    out: Dict[str, List[Entity]] = {}
    for e in _ents(text, config=config):
        out.setdefault(e.type, []).append(e)
    # keep stable order by span
    return {k: sorted(v, key=lambda e: e.span) for k, v in out.items()}

def _money_found(e: Entity, amount: float, currency: str, tol: float = 1e-6) -> bool:
    return (
        e.type == "MONEY"
        and isinstance(e.value, MoneyValue)
        and abs(e.value.amount - amount) <= tol
        and (e.value.currency == currency)
    )

def _has_money_with(text: str, amounts: List[float], currencies: List[str]) -> bool:
    """Return True if at least one MONEY entity matches any (amount,currency) in given lists."""
    ms = ents_by_type(text).get("MONEY", [])
    want = set(zip(amounts, currencies))
    for e in ms:
        for amt, cur in want:
            if _money_found(e, amt, cur):
                return True
    return False


# ======================
# MONEY: symbol & codes
# ======================
def test_money_symbol_and_code_basic():
    # symbol before amount
    assert _has_money_with("Price is $6.", amounts=[6.0], currencies=["USD"]) is True

    # 3-letter code uppercase and allowed (includes suffix k)
    assert _has_money_with("EUR 3.5k", amounts=[3500.0], currencies=["EUR"]) is True

    # parentheses negative
    es = ents_by_type("Refund (1,250) USD")["MONEY"]
    assert any(abs(e.value.amount + 1250.0) < 1e-6 for e in es)
    # USD allowed
    assert any((e.value.currency == "USD") for e in es)


# =============================
# PERCENT and NUMBER shorthand
# =============================
def test_percent_and_number_shorthand():
    es_percent = ents_by_type("Rate improved to 12.5%.")["PERCENT"]
    assert len(es_percent) == 1 and abs(es_percent[0].value.value - 12.5) < 1e-9

    # shorthand number with k/m/b should also appear as NUMBER
    es_num = ents_by_type("Capacity is 3.5k units.")["NUMBER"]
    assert any(abs(e.value.value - 3500.0) < 1e-9 for e in es_num)


# =================
# QUANTITY variants
# =================
def test_quantity_variants_units():
    txt = "Battery 500 mAh; Distance 10 km; Temp 25 °C."
    qs = ents_by_type(txt).get("QUANTITY", [])
    want = {("mAh", 500.0), ("km", 10.0), ("°C", 25.0)}
    got = {(e.value.unit, e.value.value) for e in qs}
    assert want.issubset(got)


# ===========
# DATE cases
# ===========
def test_date_variants_stable():
    # Use unambiguous formats with explicit year to avoid locale/relative issues
    cases = [
        ("July 24, 2024", "2024-07-24"),
        ("24 July 2024", "2024-07-24"),
        ("2024-07-24", "2024-07-24"),
        ("on 2024-07-24 at 10:30", "2024-07-24"),
        ("in 2024", "2024-"),  # year-only phrase → allow any month/day default
    ]
    for text, prefix in cases:
        ds = ents_by_type(text)["DATE"]
        assert len(ds) >= 1
        iso = ds[0].value.iso
        assert iso.startswith(prefix)

@pytest.mark.parametrize("text, expect_date", [
    ("The event is on 2024-01-05.", True),
    ("Event falls in 2024.", True),
    ("In the month July day of 24 the year of 2024.", True),
])
def test_date_freeform_phrases(text, expect_date):
    ds = ents_by_type(text)["DATE"]
    assert (len(ds) > 0) == expect_date


# ============================================
# Report integration & unsupported diagnostics
# ============================================
def test_report_partial_support_and_unsupported_numbers():
    q = "What are the figures?"
    a = "Q4 margin was 47% and revenue $5.4B."
    ctx = [
        "Q4 gross margin expanded to 45% on cost optimizations.",
        "FY2023 revenue printed at $5.2B according to the 10-K.",
    ]
    rep = context_utilization_report_with_entities(q, a, ctx)

    # Expect mismatches → unsupported_numbers should mention percent and money mismatches
    uns_nums = rep["unsupported_numbers"]
    # Should include PERCENT mismatch
    assert any(x.startswith("PERCENT:") for x in uns_nums)
    # And MONEY mismatch (value differs)
    assert any(x.startswith("MONEY:") for x in uns_nums)


def test_report_entity_priority_suppresses_nested_number():
    # Number inside money/date should not be double-counted as NUMBER
    txt = "Tickets cost $50 on 2024-01-05."
    by_type = ents_by_type(txt)
    # Expect MONEY and DATE present
    assert len(by_type.get("MONEY", [])) >= 1
    assert len(by_type.get("DATE", [])) >= 1
    # And no bare NUMBER for "50" or "2024" because they are covered by MONEY/DATE
    # (NUMBERs may still appear for other tokens outside those spans)
    for e in by_type.get("NUMBER", []):
        # Ensure any number isn't fully inside the MONEY or DATE spans
        spans = [x.span for x in by_type.get("MONEY", [])] + [x.span for x in by_type.get("DATE", [])]
        assert not any(e.span[0] >= s and e.span[1] <= t for (s, t) in spans)


# ================================
# Regression: punctuation & spans
# ================================
def test_money_spacing_and_suffixed_code():
    # Ensure "EUR 3.5k" parsed as MONEY with amount 3500 and EUR
    es = ents_by_type("We saw EUR 3.5k in refunds.")["MONEY"]
    assert any(_money_found(e, 3500.0, "EUR") for e in es)

    # Ensure "$2,500" works
    es2 = ents_by_type("Price was $2,500 today.")["MONEY"]
    assert any(abs(e.value.amount - 2500.0) < 1e-9 and e.value.currency == "USD" for e in es2)