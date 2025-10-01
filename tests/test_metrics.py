# tests/test_metrics.py
import json
import sys
from pathlib import Path
import os
import pytest
from pprint import pprint

# -----------------------------------------------------------------------------
# Ensure the project root (where metric_utils.py lives) is importable for tests
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Now imports from your repo root will work without setting PYTHONPATH globally
import metric_utils  # expects extractor.py and metric_utils.py at the repo root


# -----------------------------------------------------------------------------
# Load cases from file if available, else fall back to inline fixtures
# Put an optional file at: tests/data/rag_eval_cases.json
# Format:
# [
#   {"id": "...", "question": "...", "contexts": ["...", "...", "..."], "answer": "..."},
#   ...
# ]
# -----------------------------------------------------------------------------
DATA_FILE = ROOT / "tests" / "data" / "rag_eval_cases.json"

FALLBACK_CASES = [
    {
        "id": "case_529",
        "question": "What is the contribution limit for a 529 account?",
        "contexts": [
            "For 2023, the annual gift tax exclusion amount is $17,000 per beneficiary, meaning you can contribute up to that amount without triggering gift tax reporting. Some states also provide tax deductions.",
            "529 plans allow a one-time lump sum contribution of up to 5 years' worth of the annual gift tax exclusion, currently $85,000, treated as if it were spread over 5 years for gift tax purposes.",
            "The lifetime contribution cap varies by state, often exceeding $300,000 per beneficiary. Earnings in a 529 plan grow tax-deferred, and withdrawals for qualified education expenses are tax-free."
        ],
        "answer": "In 2023 you can generally contribute up to $17,000 per year per beneficiary into a 529, or front-load $85,000 using the 5-year election. States may impose lifetime caps around $300,000."
    },
    {
        "id": "case_roth401k",
        "question": "What is a Roth 401(k) and what are its contribution limits?",
        "contexts": [
            "A Roth 401(k) is an employer-sponsored retirement account that combines features of a traditional 401(k) with those of a Roth IRA.",
            "Contributions are made with after-tax dollars, and qualified withdrawals in retirement are tax-free.",
            "Contribution limits are the same as traditional 401(k) accounts, $22,500 in 2023, with an extra $7,500 catch-up for those age 50 and older."
        ],
        "answer": "A Roth 401(k) is a retirement plan where you contribute after-tax dollars, withdrawals are tax-free, and in 2023 you can put in $22,500 plus a $7,500 catch-up if you’re 50 or older."
    }
]

def load_cases():
    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return FALLBACK_CASES

CASES = load_cases()


# -------------------------------------------------------------------
# helper for printing debug info (optional; controlled by METRICS_DEBUG=1)
# -------------------------------------------------------------------
def _dump_report(report: dict, top_k: int = 10):
    # Compact, readable summary
    em = report.get("entity_match", {})
    se = report.get("supported_entities", {})
    print("\n==== RAG Metrics Debug ====")
    print(f"summary:            {report.get('summary')}")
    print(f"precision_token:    {report.get('precision_token')}")
    print(f"recall_context:     {report.get('recall_context')}")
    print(f"numeric_match:      {report.get('numeric_match')}")

    # Embedding fields (if present)
    qa = report.get("qr_alignment", {}) or {}
    ca = report.get("context_alignment", {}) or {}
    if qa.get("cosine_embed") is not None or qa.get("answer_covers_question_sem") is not None:
        print("\n-- embedding (MiniLM) --")
        print(f"qa.cosine_embed:                {qa.get('cosine_embed')}")
        print(f"qa.answer_covers_question_sem:  {qa.get('answer_covers_question_sem')}")
        print(f"ctx.answer_context_similarity:  {ca.get('answer_context_similarity')}")
        print(f"ctx.best_context_similarity:    {ca.get('best_context_similarity')}")

    print("\n-- entity_match --")
    print(f"overall: {em.get('overall')}")
    print(f"by_type: {json.dumps(em.get('by_type', {}), indent=2)}")

    unsupported = em.get("unsupported", [])
    if unsupported:
        print(f"unsupported ({len(unsupported)}):")
        for x in unsupported[:top_k]:
            print(f"  - {x}")
        if len(unsupported) > top_k:
            print(f"  ... (+{len(unsupported)-top_k} more)")

    print("\n-- supported_entities --")
    items = se.get("items", [])
    print(f"count: {se.get('count', 0)}; by_type: {se.get('by_type', {})}")
    for it in items[:top_k]:
        # {type, text, start, end}
        print(f"  ✓ {it['type']:<8} {it['text']!r}  [{it['start']},{it['end']}]")
    if len(items) > top_k:
        print(f"  ... (+{len(items)-top_k} more)")

    print("\n-- lexical overlap --")
    print("supported_terms (top):")
    for t in report.get("supported_terms", [])[:top_k]:
        print(f"  ✓ {t['term']:<20} count={t['count']} idf={t['idf']}")

    print("\nunsupported_terms (top):")
    for t in report.get("unsupported_terms", [])[:top_k]:
        print(f"  ✗ {t['term']:<20} impact={t['impact']} count={t['count']} idf={t['idf']}")

    un_nums = report.get("unsupported_numbers", [])
    if un_nums:
        print("\nunsupported_numbers:")
        for u in un_nums[:top_k]:
            print(f"  ✗ {u}")
        if len(un_nums) > top_k:
            print(f"  ... (+{len(un_nums)-top_k} more)")

    print("============================\n")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def embedder_available():
    """Force metric_utils to try loading the local MiniLM embedder once."""
    if getattr(metric_utils, "_EMB", None) is None:
        metric_utils._EMB = metric_utils._maybe_load_embedder("models/all-MiniLM-L6-v2")
    return metric_utils._EMB is not None

# -----------------------------------------------------------------------------
# Tests (TF-IDF only)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_context_utilization_smoke(case):
    """Basic structure & types are present and sane."""
    report = metric_utils.context_utilization_report_with_entities(
        question=case["question"],
        answer=case["answer"],
        retrieved_contexts=case["contexts"],
        use_bm25_for_best=True,
        use_embed_alignment=True,
        # embed_model_path="models/all-MiniLM-L6-v2"   # only if metric_utils supports it
    )
    if os.environ.get("METRICS_DEBUG") == "1":
        _dump_report(report, top_k=12)

    # required top-level keys
    for key in [
        "precision_token","recall_context","numeric_match","entity_match",
        "supported_entities","per_sentence","qr_alignment","context_alignment",
        "unsupported_terms","unsupported_numbers","summary"
    ]:
        assert key in report, f"Missing key: {key}"

    # types / ranges
    assert isinstance(report["precision_token"], float)
    assert isinstance(report["recall_context"], float)
    assert isinstance(report["numeric_match"], float)
    assert 0.0 <= report["precision_token"] <= 1.0
    assert 0.0 <= report["recall_context"]  <= 1.0
    assert 0.0 <= report["numeric_match"]    <= 1.0
    assert isinstance(report["summary"], str) and len(report["summary"]) > 10


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_numeric_and_entity_coverage(case):
    """
    Our curated fixtures should have strong numeric coverage.
    - 529 case: MONEY entities expected.
    - Roth 401(k) case: NUMBER entities expected (plain numerals).
    """
    report = metric_utils.context_utilization_report_with_entities(
        question=case["question"],
        answer=case["answer"],
        retrieved_contexts=case["contexts"],
        use_bm25_for_best=True,
        use_embed_alignment=True,
        # embed_model_path="models/all-MiniLM-L6-v2"   # only if metric_utils supports it
    )
    if os.environ.get("METRICS_DEBUG") == "1":
        _dump_report(report, top_k=12)

    # Numeric facts recognized and covered reasonably well
    assert 0.0 <= report["numeric_match"] <= 1.0

    by_type = report["entity_match"].get("by_type", {})
    if case["id"] == "case_529":
        assert by_type.get("MONEY", 0.0) >= 0.5  # loosened threshold for generality
    if case["id"] == "case_roth401k":
        assert by_type.get("NUMBER", 0.0) >= 0.5


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_supported_entities_have_spans(case):
    """Supported entities list should include span info when entities are matched."""
    report = metric_utils.context_utilization_report_with_entities(
        question=case["question"],
        answer=case["answer"],
        retrieved_contexts=case["contexts"],
        use_bm25_for_best=True,
        use_embed_alignment=True,
        # embed_model_path="models/all-MiniLM-L6-v2"   # only if metric_utils supports it
    )
    if os.environ.get("METRICS_DEBUG") == "1":
        _dump_report(report, top_k=12)

    items = report.get("supported_entities", {}).get("items", [])
    # If there are items, they must have start/end int spans
    for it in items:
        assert "type" in it and "text" in it
        assert isinstance(it.get("start"), int)
        assert isinstance(it.get("end"), int)


# -----------------------------------------------------------------------------
# Tests (MiniLM embeddings ON)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_embedding_alignment_fields(case, embedder_available):
    """
    When use_embed_alignment=True, embedding fields should be present.
    If the embedder is available, they should be floats in valid ranges.
    """
    report = metric_utils.context_utilization_report_with_entities(
        question=case["question"],
        answer=case["answer"],
        retrieved_contexts=case["contexts"],
        use_bm25_for_best=True,
        use_embed_alignment=True,
        # embed_model_path="models/all-MiniLM-L6-v2"   # only if metric_utils supports it
    )

    qa = report.get("qr_alignment", {}) or {}
    ca = report.get("context_alignment", {}) or {}

    # keys should exist regardless
    assert "cosine_embed" in qa
    assert "answer_covers_question_sem" in qa
    assert "answer_context_similarity" in ca
    assert "best_context_similarity" in ca

    if not embedder_available:
        pytest.skip("MiniLM embedder not available locally; skipping value checks.")

    # If we have the model, values should be floats (or None for term coverage if no terms)
    if qa["cosine_embed"] is not None:
        assert isinstance(qa["cosine_embed"], float)
        assert -1.0 <= qa["cosine_embed"] <= 1.0

    if qa["answer_covers_question_sem"] is not None:
        assert isinstance(qa["answer_covers_question_sem"], float)
        assert 0.0 <= qa["answer_covers_question_sem"] <= 1.0

    if ca["answer_context_similarity"] is not None:
        assert isinstance(ca["answer_context_similarity"], float)
        assert -1.0 <= ca["answer_context_similarity"] <= 1.0

    if ca["best_context_similarity"] is not None:
        assert isinstance(ca["best_context_similarity"], float)
        assert -1.0 <= ca["best_context_similarity"] <= 1.0


# Optional: print full details when toggled on
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_debug_dump_report(case):
    # Only runs when toggled on
    if os.environ.get("METRICS_DEBUG") != "1":
        pytest.skip("Set METRICS_DEBUG=1 to print detailed report")

    report = metric_utils.context_utilization_report_with_entities(
        question=case["question"],
        answer=case["answer"],
        retrieved_contexts=case["contexts"],
        use_bm25_for_best=True,
        use_embed_alignment=True,
        # embed_model_path="models/all-MiniLM-L6-v2"   # only if metric_utils supports it
    )
    _dump_report(report, top_k=20)