# tests/test_context_entity_matching.py

# near the top of the file
import pytest
from dataclasses import replace

from extractor import (
    ExtractorConfig,
    extract_entities,
    extract_by_type,
    entity_coverage,
    match_entity_values,
)

# Try to pull a shared default ExtractorConfig if available; fall back to local defaults.
try:
    # Preferred name in shared_config
    from shared_config import DEFAULT_EXTRACTOR_CONFIG as _DEFAULT_EXTRACTOR_CONFIG  # type: ignore
except Exception:
    try:
        # Back-compat alias some repos use
        from shared_config import DEFAULT_CONFIG as _DEFAULT_EXTRACTOR_CONFIG  # type: ignore
    except Exception:
        _DEFAULT_EXTRACTOR_CONFIG = None  # type: ignore

# ---- spaCy availability check (so the suite is portable) ----
try:
    import spacy  # type: ignore
    try:
        _ = spacy.load("en_core_web_sm")
        HAS_SPACY = True
    except Exception:
        HAS_SPACY = False
except Exception:
    HAS_SPACY = False

# Turn spaCy fusion ON for these end-to-end context/answer tests.
# If a project-wide default exists in shared_config, start from it and override what we need;
# otherwise, fall back to a local config.
if _DEFAULT_EXTRACTOR_CONFIG is not None:
    CFG = replace(_DEFAULT_EXTRACTOR_CONFIG, use_spacy_fusion=True, timezone="UTC")
else:
    CFG = ExtractorConfig(use_spacy_fusion=True, timezone="UTC")

def ents_by_type(text, cfg=CFG):
    return extract_by_type(text, config=cfg)


def first(ents_by_t, t):
    xs = ents_by_t.get(t, [])
    return xs[0] if xs else None

# --- 1) DATE value match across different surface forms ---
@pytest.mark.skipif(not HAS_SPACY, reason="spaCy en_core_web_sm required for DATE fusion")
def test_date_value_match_across_formats():
    context = "The launch is on 24th of July 2024."
    answer  = "Confirmed date: 2024-07-24."

    ctx = ents_by_type(context)
    ans = ents_by_type(answer)

    c_date = first(ctx, "DATE")
    a_date = first(ans, "DATE")

    assert c_date and a_date
    # Spans are different but values should normalize to same ISO day
    assert c_date.span != a_date.span
    assert match_entity_values(a_date, c_date) is True

    cov = entity_coverage([a_date], [c_date])
    assert cov["overall"] == 1.0
    assert cov["by_type"]["DATE"] == 1.0

@pytest.mark.skipif(not HAS_SPACY, reason="spaCy en_core_web_sm required for DATE fusion")
def test_date_value_match_july_24_vs_iso():
    context = "Event happens July 24, 2024."
    answer  = "Happened on 2024-07-24."
    c_date = first(ents_by_type(context), "DATE")
    a_date = first(ents_by_type(answer), "DATE")
    assert c_date and a_date
    assert c_date.span != a_date.span
    assert match_entity_values(a_date, c_date) is True

# --- 2) MONEY: words vs symbol ---
def test_money_words_vs_symbol():
    context = "two hundred dollars were refunded."
    answer  = "Refund was $200."
    c_money = first(ents_by_type(context), "MONEY")
    a_money = first(ents_by_type(answer), "MONEY")
    assert c_money and a_money
    assert c_money.span != a_money.span
    assert match_entity_values(a_money, c_money) is True

    cov = entity_coverage([a_money], [c_money])
    assert cov["overall"] == 1.0
    assert cov["by_type"]["MONEY"] == 1.0

# --- 3) PERCENT: parentheses negative vs minus sign ---
def test_percent_parens_vs_minus():
    context = "Margin was (12.5%)."
    answer  = "Margin = -12.5%."
    c_pct = first(ents_by_type(context), "PERCENT")
    a_pct = first(ents_by_type(answer), "PERCENT")
    assert c_pct and a_pct
    assert c_pct.span != a_pct.span
    assert match_entity_values(a_pct, c_pct) is True

# --- 4) NUMBER: words vs digits ---
def test_number_words_vs_numeric():
    context = "We shipped twenty-one units."
    answer  = "We shipped 21 units."
    c_num = first(ents_by_type(context), "NUMBER")
    a_num = first(ents_by_type(answer), "NUMBER")
    assert c_num and a_num
    assert c_num.span != a_num.span
    assert match_entity_values(a_num, c_num) is True

# --- 5) QUANTITY: unit canonicalization (μm vs µm forms) ---
def test_quantity_unit_canonicalization_micro_m():
    context = "Feature size is 2μm."   # Greek mu
    answer  = "Feature size is 2 µm."  # micro sign
    c_qty = first(ents_by_type(context), "QUANTITY")
    a_qty = first(ents_by_type(answer), "QUANTITY")
    assert c_qty and a_qty
    assert c_qty.span != a_qty.span
    assert match_entity_values(a_qty, c_qty) is True

# --- 6) Mixed bag: one match, one mismatch -> overall fractional coverage ---
@pytest.mark.skipif(not HAS_SPACY, reason="spaCy en_core_web_sm required for DATE fusion")
def test_mixed_entities_partial_coverage():
    context = "On 2024-07-24 we paid $1,200."
    answer  = "On July 24, 2024 we paid $1,250."
    ctx_ents = extract_entities(context, config=CFG)
    ans_ents = extract_entities(answer, config=CFG)

    # Pull the first date and money from each
    c_date = next(e for e in ctx_ents if e.type == "DATE")
    c_money = next(e for e in ctx_ents if e.type == "MONEY")
    a_date = next(e for e in ans_ents if e.type == "DATE")
    a_money = next(e for e in ans_ents if e.type == "MONEY")

    # Dates should match; money should not
    assert match_entity_values(a_date, c_date) is True
    assert match_entity_values(a_money, c_money) is False

    cov = entity_coverage([a_date, a_money], [c_date, c_money])
    # By type expectations
    assert cov["by_type"]["DATE"] == 1.0
    assert cov["by_type"]["MONEY"] == 0.0
    # Overall = 1 hit / 2 answer entities
    assert abs(cov["overall"] - 0.5) < 1e-9

# --- 7) Span vs value precedence: spans differ but values equal must match ---
def test_span_irrelevant_when_value_equal():
    context = "The fee is USD 200."
    answer  = "The fee was $200 today."
    c_money = first(ents_by_type(context), "MONEY")
    a_money = first(ents_by_type(answer), "MONEY")
    assert c_money and a_money
    # Different spans and different surface forms ($ vs USD code)
    assert c_money.span != a_money.span
    assert match_entity_values(a_money, c_money) is True

# --- 8) Sanity: types must match to be counted ---
def test_type_mismatch_does_not_match():
    # NUMBER 200 vs MONEY $200 should not match
    context = "The amount is 200."
    answer  = "The amount is $200."
    c_num = first(ents_by_type(context), "NUMBER")
    a_money = first(ents_by_type(answer), "MONEY")
    assert c_num and a_money
    assert c_num.type != a_money.type
    assert match_entity_values(a_money, c_num) is False

    cov = entity_coverage([a_money], [c_num])
    # No MONEY in context; overall 0
    assert cov["overall"] == 0.0
    # By-type only reports types present in the answer; MONEY has 0 matched
    assert cov["by_type"].get("MONEY", 0.0) == 0.0

# --- Additional robustness tests for context/answer matching ---

@pytest.mark.skipif(not HAS_SPACY, reason="spaCy en_core_web_sm required for DATE fusion")
def test_date_relative_with_fixed_base():
    """Relative phrase should normalize to the same absolute day given a fixed base."""
    cfg = ExtractorConfig(use_spacy_fusion=True, timezone="UTC")
    # Choose a stable base: 2024-07-17 is a Wednesday; next Friday => 2024-07-19
    from datetime import datetime
    base = datetime(2024, 7, 17)

    context = "Let's meet next Friday."
    answer  = "Let's meet on 2024-07-19."

    c_ents = extract_by_type(context, config=cfg, ref_date=base)
    a_ents = extract_by_type(answer,  config=cfg, ref_date=base)

    c_date = c_ents.get("DATE", [None])[0]
    a_date = a_ents.get("DATE", [None])[0]

    assert c_date and a_date
    assert c_date.span != a_date.span
    assert match_entity_values(a_date, c_date) is True


def test_money_negative_equivalence():
    """Parens-negative vs leading-minus should be considered the same MONEY value."""
    context = "Change was (1,200) USD."
    answer  = "Change was -$1,200."
    c_money = first(ents_by_type(context), "MONEY")
    a_money = first(ents_by_type(answer),  "MONEY")
    assert c_money and a_money
    assert match_entity_values(a_money, c_money) is True


def test_money_duplicate_context_single_answer_coverage():
    """If context has duplicates but answer has one, we still count one match only."""
    context = "We paid $200 and another $200 yesterday."
    answer  = "$200 was paid."

    ctx_ents = extract_entities(context, config=CFG)
    ans_ents = extract_entities(answer,  config=CFG)

    # answer has one MONEY
    a_money = [e for e in ans_ents if e.type == "MONEY"]
    c_money = [e for e in ctx_ents if e.type == "MONEY"]
    assert len(a_money) == 1 and len(c_money) >= 2

    cov = entity_coverage(a_money, c_money)
    assert cov["by_type"]["MONEY"] == 1.0
    assert cov["overall"] == 1.0


def test_money_currency_mismatch_is_not_match():
    """Same amount but different currencies should not match."""
    context = "We paid $200."
    answer  = "Total was EUR 200."
    c_money = first(ents_by_type(context), "MONEY")
    a_money = first(ents_by_type(answer),  "MONEY")
    assert c_money and a_money
    assert match_entity_values(a_money, c_money) is False


def test_money_space_grouping_vs_commas():
    """1 200 with a space should normalize to the same value as 1,200."""
    context = "Invoice: USD 1 200."
    answer  = "Invoice: $1,200."
    c_money = first(ents_by_type(context), "MONEY")
    a_money = first(ents_by_type(answer),  "MONEY")
    assert c_money and a_money
    assert match_entity_values(a_money, c_money) is True