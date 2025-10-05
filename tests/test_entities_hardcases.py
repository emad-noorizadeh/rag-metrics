# tests/test_entities_hardcases.py
import re
import pytest

import extractor
from extractor import (
    Entity,
    MoneyValue,
    DateValue,
    ExtractorConfig,
)

# Use spaCy fusion so we benefit from NER + deterministic parsing
CFG = ExtractorConfig(use_spacy_fusion=True)

# ---------- helpers ----------
def ents_by_type(text: str, cfg: ExtractorConfig = CFG):
    out = {}
    for e in extractor.extract_entities(text, config=cfg):
        out.setdefault(e.type, []).append(e)
    # stable order
    return {k: sorted(v, key=lambda e: e.span) for k, v in out.items()}

def money_found(e: Entity, amt: float, cur: str, tol=1e-6) -> bool:
    return (
        e.type == "MONEY"
        and isinstance(e.value, MoneyValue)
        and (e.value.currency == cur)
        and abs(e.value.amount - amt) <= tol
    )

# ---------- MONEY hard cases ----------

@pytest.mark.parametrize("text, amt, cur", [
    # symbols + separators
    ("Price was $2,500 today.", 2500.0, "USD"),
    ("List: €1.234,56 is not our format, but €1,234.56 is.", 1234.56, "EUR"),
    ("Micropayments like £0.99 are supported.", 0.99, "GBP"),
    # 3-letter codes before amount
    ("We booked USD 50 as a fee.", 50.0, "USD"),
    ("Refunds total EUR 3.5k for the quarter.", 3500.0, "EUR"),
    ("CAPEX was CAD -1.2M last year.", -1_200_000.0, "CAD"),
    ("OPEX hit JPY 12,300.", 12300.0, "JPY"),
    # symbol + big suffix
    ("Pipeline was $5b by YE.", 5_000_000_000.0, "USD"),
])
def test_money_diverse_formats(text, amt, cur):
    ms = ents_by_type(text).get("MONEY", [])
    assert any(money_found(e, amt, cur) for e in ms), f"MONEY not found for {text!r}"

def test_money_parentheses_negative_and_spacing():
    # parentheses negative with symbol
    ms1 = ents_by_type("Change was $(1,250) this month.").get("MONEY", [])
    assert any(money_found(e, -1250.0, "USD") for e in ms1)

    # parentheses negative with code-before-amount
    ms2 = ents_by_type("Refunds were USD (1,250).").get("MONEY", [])
    assert any(money_found(e, -1250.0, "USD") for e in ms2)

def test_money_no_double_count_number_inside_money():
    by = ents_by_type("Tickets cost $50 and VAT is 12%.")
    # $50 should be MONEY; "50" should not also appear as bare NUMBER inside same span
    assert len(by.get("MONEY", [])) >= 1
    money_spans = [e.span for e in by.get("MONEY", [])]
    for n in by.get("NUMBER", []):
        assert not any(n.span[0] >= s and n.span[1] <= t for (s, t) in money_spans)

# Cases that now pass with our current extractor improvements
@pytest.mark.parametrize("text", [
    "It cost one dollar.",
    "two hundred dollars were refunded.",
    "We owe three dollars.",
])
def test_money_word_numbers_minimal(text):
    ms = ents_by_type(text).get("MONEY", [])
    assert len(ms) >= 1

# Still expected to fail until we fully normalize word-numbers with currencies
@pytest.mark.xfail(reason="Word-number amounts not fully normalized to numeric MONEY yet")
@pytest.mark.parametrize("text", [
    "They paid two euros.",
    "Roughly five thousand USD was spent.",
    "They charged twenty one euro for shipping.",
])
def test_money_word_numbers_future(text):
    ms = ents_by_type(text).get("MONEY", [])
    assert len(ms) >= 1

# ---------- WORD-NUMBER -> NUMBER (span + normalization) ----------

# Enable spaCy for NUMBER so we get tight spans from CARDINAL/ORDINAL mentions
CFG_NUM = ExtractorConfig(use_spacy_fusion=True)
CFG_NUM.use_spacy_for["NUMBER"] = True

@pytest.mark.parametrize("text, expected", [
    ("We saw three of them.", 3.0),
    ("About two hundred people came.", 200.0),
    ("Top five results were relevant.", 5.0),
    ("Only twenty-one tickets remain.", 21.0),
])
def test_cardinal_words_become_number_with_span(text, expected):
    ds = ents_by_type(text, cfg=CFG_NUM).get("NUMBER", [])
    # We expect at least one NUMBER with the correct normalized value.
    assert any(abs(e.value.value - expected) < 1e-9 for e in ds), f"NUMBER {expected} not found in: {text!r}"
    # And at least one such NUMBER should have a reasonably tight span (<= length of the input and not the whole sentence from index 0).
    # This checks that we're likely using spaCy's entity span rather than the entire sentence.
    assert any(e.span != (0, len(text)) for e in ds), "Expected a tight NUMBER span (not the whole sentence)"

# ---------- DATE hard cases ----------

@pytest.mark.parametrize("text,prefix", [
    # ISO & explicit forms
    ("Event is on 2024-07-24.", "2024-07-24"),
    ("on 2024-07-24 at 10:30", "2024-07-24"),
    ("July 24, 2024", "2024-07-24"),
    ("24 July 2024", "2024-07-24"),
    ("24th of July 2024", "2024-07-24"),
    # month-year / year-only (we just assert prefix)
    ("Sometime in July 2024.", "2024-07-"),
    ("In 2024 we expand.", "2024-"),
])
def test_date_diverse_formats(text, prefix):
    ds = ents_by_type(text).get("DATE", [])
    assert len(ds) >= 1
    iso = ds[0].value.iso
    assert iso.startswith(prefix)

@pytest.mark.parametrize("text", [
    "Meeting next Friday.",
    "Shipment arrives tomorrow.",
    "We closed last Monday.",
])
def test_date_relative_present(text):
    # For relative phrasing, just require that at least one DATE is detected
    ds = ents_by_type(text).get("DATE", [])
    assert len(ds) >= 1

def test_date_and_money_together_priority_and_no_leak():
    text = "Go-live is 2024-01-05 and the fee is $1,200."
    by = ents_by_type(text)
    assert len(by.get("DATE", [])) >= 1
    assert any(money_found(e, 1200.0, "USD") for e in by.get("MONEY", []))
    # No NUMBER duplicates from inside the money span
    money_spans = [e.span for e in by.get("MONEY", [])]
    for n in by.get("NUMBER", []):
        assert not any(n.span[0] >= s and n.span[1] <= t for (s, t) in money_spans)

# ---- Additional edge cases: MONEY, PERCENT, DATE, NUMBER, QUANTITY, FUSION, GUARDRAILS ----

# 1) MONEY: locale, spacing, suffixes, and false positives
@pytest.mark.parametrize("text, amt, cur", [
    # symbol + suffix with space
    ("Raised $ 5m in funding.", 5_000_000.0, "USD"),
    # uppercase suffix
    ("Budget was $12M this year.", 12_000_000.0, "USD"),
    # code-after-amount with suffix
    ("We spent 2.5k USD on ads.", 2_500.0, "USD"),
])
def test_money_spacing_suffix_variants(text, amt, cur):
    ms = ents_by_type(text).get("MONEY", [])
    assert any(money_found(e, amt, cur) for e in ms)


def test_money_whitelist_blocks_false_codes():
    # Should NOT match as MONEY: "AND 5" (AND is not a currency code)
    ms = ents_by_type("We saw AND 5 reported.").get("MONEY", [])
    assert not ms


def test_money_symbol_trailing_is_not_supported_yet():
    # Many locales use trailing symbol (e.g., 5€). We don't support it (by design).
    ms = ents_by_type("It costs 5€ today.").get("MONEY", [])
    assert ms == []  # explicit non-support to catch regressions


# 2) PERCENT: negatives with parentheses & no number-leak

def test_percent_parens_negative_and_no_number_leak():
    by = ents_by_type("Margin was (12.5%) QoQ.")
    ps = by.get("PERCENT", [])
    assert any(abs(p.value.value + 12.5) < 1e-9 for p in ps)
    # Ensure the 12.5 inside % is not also a bare NUMBER
    pct_spans = [e.span for e in ps]
    for n in by.get("NUMBER", []):
        assert not any(n.span[0] >= s and n.span[1] <= t for (s, t) in pct_spans)


# 3) DATE: ranges, abbreviated years, and "week" phrases

def test_date_range_two_dates_present():
    ds = ents_by_type("Between 2024-01-05 and 2024-01-10 we implemented changes.").get("DATE", [])
    # Accept at least two dates
    assert len(ds) >= 2


@pytest.mark.xfail(reason="Abbrev years ('\u201924) not normalized yet")
def test_date_abbreviated_year_tick():
    ds = ents_by_type("Launched on Sept 7, ’24.").get("DATE", [])
    assert len(ds) >= 1
    assert ds[0].value.iso.startswith("2024-09-07")


def test_date_relative_week_phrases():
    ds = ents_by_type("We ship in two weeks.").get("DATE", [])
    assert len(ds) >= 1  # spaCy usually tags as DATE; our fallback may also catch


# 4) NUMBER (word-cardinals): conjunctions, hyphens, big numbers
@pytest.mark.parametrize("text, expected", [
    ("one hundred and five users", 105.0),
    ("twenty-one issues closed", 21.0),
    ("three thousand two hundred", 3200.0),
])
def test_cardinal_words_more_variants(text, expected):
    ds = ents_by_type(text, cfg=CFG_NUM).get("NUMBER", [])
    assert any(abs(e.value.value - expected) < 1e-9 for e in ds)
    assert any(e.span != (0, len(text)) for e in ds)  # tight span


# 5) QUANTITY: tight coverage of units and symbol variations
@pytest.mark.parametrize("text, unit, val", [
    ("Feature size is 2 \u00B5m.", "\u00B5m", 2.0),
    ("Pixel pitch at 2\u03BCm.", "\u00B5m", 2.0),         # Greek mu
    ("Thermal limit is 25\u00B0C.", "\u00B0C", 25.0),     # no space
    ("Battery is 1.5 kWh.", "kWh", 1.5),
    ("Camera uses 500mAh cells.", "mAh", 500.0) # no space
])
def test_quantity_unit_variants(text, unit, val):
    qs = ents_by_type(text).get("QUANTITY", [])
    assert any(q.value.unit == unit and abs(q.value.value - val) < 1e-9 for q in qs)


# 6) Multi-entity overlaps and ordering

def test_multiple_entities_order_and_overlap_rules():
    txt = "On 2024-07-24 we paid EUR 3.5k and USD 200."
    by = ents_by_type(txt)
    assert len(by.get("DATE", [])) >= 1
    assert any(money_found(e, 3500.0, "EUR") for e in by.get("MONEY", []))
    assert any(money_found(e, 200.0, "USD") for e in by.get("MONEY", []))
    # Number inside MONEY not duplicated
    money_spans = [e.span for e in by.get("MONEY", [])]
    for n in by.get("NUMBER", []):
        assert not any(n.span[0] >= s and n.span[1] <= t for (s, t) in money_spans)


# 7) Fusion behavior sanity checks

def test_fusion_picks_spacy_when_det_has_none_for_relative():
    cfg = ExtractorConfig(use_spacy_fusion=True)
    ds = ents_by_type("See you tomorrow.", cfg).get("DATE", [])
    assert len(ds) >= 1  # spaCy adds it; det path would miss it


def test_deterministic_still_catches_iso_without_spacy():
    cfg = ExtractorConfig(use_spacy_fusion=False)
    ds = ents_by_type("Event: 2024-07-24.", cfg).get("DATE", [])
    assert len(ds) >= 1
    assert ds[0].value.iso.startswith("2024-07-24")


# 8) Known-unsupported (guardrails)
@pytest.mark.xfail(reason="We don't currently parse 'bn'/'mm' finance suffixes")
@pytest.mark.parametrize("text", [
    "Revenue topped $5bn.",
    "Capex was $200mm.",
])
def test_money_finance_suffixes_not_supported_yet(text):
    assert ents_by_type(text).get("MONEY", [])


@pytest.mark.xfail(reason="Trailing currency symbols not supported")
def test_money_trailing_symbol_not_supported():
    assert ents_by_type("Price is 199€.").get("MONEY", [])