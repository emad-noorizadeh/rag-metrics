# Emad Noorizadeh
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Metric Utilities for RAG System

This module provides comprehensive metric calculation functions for evaluating RAG (Retrieval-Augmented Generation) 
system performance. It includes context utilization analysis, confidence scoring, faithfulness measurement, 
and completeness assessment.

Key Features:
- Context utilization calculation based on word overlap between answers and retrieved context
- Confidence scoring using heuristics based on answer quality and context relevance
- Faithfulness measurement to ensure answers are grounded in retrieved context
- Completeness assessment to evaluate how fully questions are answered
- Robust edge case handling for null/empty inputs and clarification scenarios

Author: Emad Noorizadeh
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from math import log, sqrt

# NEW imports from typed extractor
from extractor import (
    ExtractorConfig, DEFAULT_CONFIG,
    Entity, EntityType,
    extract_entities, extract_by_type,
    is_money, is_number, is_percent, is_date, is_quantity, is_phone,
    money_equal, number_equal, date_equal, entity_coverage,
    canonicalize_date, match_entity_values,   # <-- ensure this is included
)

# -------------------------
# Tokenization / weighting
# -------------------------

_STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
    'is','are','was','were','be','been','being','have','has','had','do','does','did',
    'will','would','could','should','may','might','must','can','this','that','these','those',
    'i','you','he','she','it','we','they','me','him','her','us','them','my','your',
    'his','hers','its','our','their','as','from','about','into','over','under','than',
    'then','so','if','not','no','yes','also','just','only','very','more','most','such'
}

_WORD_RE = re.compile(r"\b[\w$%.-]+\b", re.UNICODE)

def _simple_lemma(tok: str) -> str:
    """Cheap, deterministic normalizer w/out external deps."""
    t = tok.lower()
    # strip punctuation-like tails
    t = t.strip(".,;:!?()[]{}'\"")
    # normalize money like $20,000.00 -> $20000
    if t.startswith("$"):
        digits = re.sub(r"[^\d]", "", t[1:])
        return f"${digits}" if digits else "$"
    # normalize percents like 12.5% -> 12.5%
    if t.endswith("%"):
        core = t[:-1]
        core = re.sub(r"[^\d.]", "", core)
        return f"{core}%" if core else "%"
    # normalize plain numbers 1,234.00 -> 1234
    if re.fullmatch(r"[-+]?\d[\d,]*\.?\d*", t):
        return re.sub(r"[,\s]", "", t)
    # crude plural/verb endings
    for suf in ("'s","'s","s","es","ed","ing"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t

def _tokens(text: str):
    return [_simple_lemma(t) for t in _WORD_RE.findall(text or "")]

def _token_spans(text: str):
    """Return (token_text, start, end) using the same regex used for tokenization."""
    for m in _WORD_RE.finditer(text or ""):
        tok = m.group(0)
        start, end = m.span()
        yield tok, start, end

def _normalized_token_spans(text: str):
    """Yield (norm_token, start, end, surface) for non-stopword informative tokens."""
    for tok, s, e in _token_spans(text):
        norm = _simple_lemma(tok)
        if not norm or norm in _STOPWORDS or norm.isdigit() or norm in {"$", "%"}:
            continue
        yield norm, s, e, tok

def _informative_terms(tokens):
    return [t for t in tokens if t and t not in _STOPWORDS and not t.isdigit() and t not in {"$", "%"}]

def _build_idf(snippets_tokens):
    """IDF from snippets; small, deterministic."""
    df = Counter()
    N = len(snippets_tokens)
    for toks in snippets_tokens:
        df.update(set(toks))
    idf = {}
    for term, d in df.items():
        # add-1 smoothing to avoid div-by-zero and keep small collections stable
        idf[term] = log((N + 1) / (d + 1)) + 1.0
    return idf

def _weighted_recall(numer: Counter, denom: Counter, idf):
    """Sum IDF over intersection divided by sum IDF over denom (denom defines 'what should be covered')."""
    w_inter = 0.0
    for t, c in numer.items():
        if t in denom:
            w_inter += idf.get(t, 1.0) * min(c, denom[t])
    w_denom = sum(idf.get(t, 1.0) * c for t, c in denom.items())
    return (w_inter / w_denom) if w_denom > 0 else 0.0

def _weighted_precision(answer_terms: List[str], context_terms: List[str], idf):
    a = Counter(answer_terms)
    c = Counter(context_terms)
    # precision: overlap against answer mass
    w_inter = 0.0
    w_answer = sum(idf.get(t, 1.0) * cnt for t, cnt in a.items())
    for t, cnt in a.items():
        if t in c:
            w_inter += idf.get(t, 1.0) * min(cnt, c[t])
    return (w_inter / w_answer) if w_answer > 0 else 0.0

# -------------------------
# Numeric coverage (typed)
# -------------------------

def _numeric_match_only(answer: str, contexts: List[str], config: ExtractorConfig = DEFAULT_CONFIG) -> float:
    """
    Fraction of numeric facts (MONEY/NUMBER/PERCENT) in answer present in any context.
    Uses typed extractor (no spaCy/quantulum3).
    """
    target_types = {"MONEY", "NUMBER", "PERCENT"}
    ans = [e for e in extract_entities(answer or "", config=config) if e.type in target_types]
    if not ans:
        return 1.0

    ctx_all: List[Entity] = []
    for s in contexts or []:
        ctx_all.extend(extract_entities(s or "", config=config))
    ctx_nums = [e for e in ctx_all if e.type in target_types]

    used = [False] * len(ctx_nums)
    hits = 0
    for ae in ans:
        for i, ce in enumerate(ctx_nums):
            if used[i] or ae.type != ce.type:
                continue
            if ae.type == "MONEY" and is_money(ae) and is_money(ce) and money_equal(ae.value, ce.value):
                used[i] = True; hits += 1; break
            if ae.type == "NUMBER" and is_number(ae) and is_number(ce) and number_equal(ae.value, ce.value):
                used[i] = True; hits += 1; break
            if ae.type == "PERCENT" and is_percent(ae) and is_percent(ce) and abs(ae.value.value - ce.value.value) <= max(1e-9, 0.0001 * max(abs(ae.value.value), abs(ce.value.value))):
                used[i] = True; hits += 1; break
    return hits / len(ans)

def _unsupported_numbers(answer: str, contexts: List[str], config: ExtractorConfig = DEFAULT_CONFIG) -> List[str]:
    """
    Return the numeric entities (MONEY/NUMBER/PERCENT) in answer that are NOT present in contexts.
    """
    target_types = {"MONEY", "NUMBER", "PERCENT"}
    ans = [e for e in extract_entities(answer or "", config=config) if e.type in target_types]
    ctx = []
    for s in contexts or []:
        ctx.extend([e for e in extract_entities(s or "", config=config) if e.type in target_types])

    used = [False] * len(ctx)
    out = []
    for ae in ans:
        matched = False
        for i, ce in enumerate(ctx):
            if used[i] or ae.type != ce.type:
                continue
            if ae.type == "MONEY" and is_money(ae) and is_money(ce) and money_equal(ae.value, ce.value):
                used[i] = True; matched = True; break
            if ae.type == "NUMBER" and is_number(ae) and is_number(ce) and number_equal(ae.value, ce.value):
                used[i] = True; matched = True; break
            if ae.type == "PERCENT" and is_percent(ae) and is_percent(ce) and abs(ae.value.value - ce.value.value) <= max(1e-9, 0.0001 * max(abs(ae.value.value), abs(ce.value.value))):
                used[i] = True; matched = True; break
        if not matched:
            if is_money(ae):       val = f"{ae.value.amount}:{ae.value.currency or ''}"
            elif is_number(ae):    val = f"{ae.value.value}"
            elif is_percent(ae):   val = f"{ae.value.value}%"
            else:                  val = ae.text
            out.append(f"{ae.type}:{val}")
    return out


# -------------------------
# Sentence & term helpers
# -------------------------

def _split_sentences(text: str):
    # Deterministic, simple splitter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", text.strip() or "")
    return [p for p in parts if p]

def _collect_supported_terms(answer: str, ctx_terms_all: List[str], idf: Dict[str, float]):
    """Collect supported terms with character spans for UI highlighting."""
    ctx_set = set(ctx_terms_all)
    counts = Counter()
    spans_per_sentence = []
    for sent in _split_sentences(answer):
        per_sent = []
        offset = answer.find(sent)
        for norm, s, e, surface in _normalized_token_spans(sent):
            if norm in ctx_set:
                counts[norm] += 1
                per_sent.append({"term": norm, "start": offset + s, "end": offset + e})
        spans_per_sentence.append({"sentence": sent, "supported_terms": per_sent})

    supported_global = [
        {"term": t, "count": c, "idf": round(idf.get(t, 1.0), 4)}
        for t, c in sorted(counts.items(), key=lambda kv: (-kv[1]*idf.get(kv[0],1.0), kv[0]))
    ]
    return supported_global, spans_per_sentence

# -------------------------
# BM25 for best context
# -------------------------

def _bm25_score(query_ctr: Counter, doc_ctr: Counter, idf: Dict[str, float], avgdl: float, dl: float,
                k1: float = 1.2, b: float = 0.75) -> float:
    score = 0.0
    for term in query_ctr.keys():
        if term not in doc_ctr:
            continue
        tf = doc_ctr[term]
        denom = tf + k1 * (1 - b + b * (dl / avgdl if avgdl > 0 else 1.0))
        score += idf.get(term, 1.0) * tf * (k1 + 1) / (denom if denom > 0 else 1.0)
    return score

def _pick_best_context_by_bm25(ans_terms: List[str], contexts_terms: List[List[str]], idf: Dict[str, float]) -> int:
    doc_ctrs = [Counter(t) for t in contexts_terms]
    ans_ctr = Counter(ans_terms)
    lengths = [sum(c.values()) for c in doc_ctrs]
    avgdl = sum(lengths) / max(1, len(lengths))
    best_i, best_s = 0, float("-inf")
    for i, doc_ctr in enumerate(doc_ctrs):
        s = _bm25_score(ans_ctr, doc_ctr, idf, avgdl, sum(doc_ctr.values()))
        if s > best_s:
            best_s, best_i = s, i
    return best_i

# -------------------------
# Typed entity alignment
# -------------------------

def _entity_alignment(answer: str, contexts: List[str], config: ExtractorConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Typed entity alignment using extractor.py (regex money/number/percent/date/quantity/phone).
    Returns match (overall/by_type/unsupported) and supported_entities with spans.
    """
    ans_ents: List[Entity] = extract_entities(answer or "", config=config)
    ctx_ents: List[Entity] = []
    for s in contexts or []:
        ctx_ents.extend(extract_entities(s or "", config=config))

    if not ans_ents:
        return {
            "match": {"overall": 1.0, "by_type": {}, "unsupported": []},
            "supported_entities": {"items": [], "by_type": {}, "count": 0}
        }

    by_type_ctx: Dict[EntityType, List[Entity]] = {}
    for e in ctx_ents:
        by_type_ctx.setdefault(e.type, []).append(e)

    total_by_type, covered_by_type = Counter(), Counter()
    supported_items, unsupported = [], []

    for ae in ans_ents:
        total_by_type[ae.type] += 1
        candidates = by_type_ctx.get(ae.type, [])
        found_i = -1
        for i, ce in enumerate(candidates):
            if match_entity_values(ae, ce):
                found_i = i
                break
        if found_i >= 0:
            covered_by_type[ae.type] += 1
            supported_items.append({"type": ae.type, "text": ae.text, "start": ae.span[0], "end": ae.span[1]})
            candidates.pop(found_i)
        else:
            if is_money(ae):       val = f"{ae.value.amount}:{ae.value.currency or ''}"
            elif is_number(ae):    val = f"{ae.value.value}"
            elif is_percent(ae):   val = f"{ae.value.value}%"
            elif is_date(ae):      val = f"{ae.value.iso}"
            elif is_quantity(ae):  val = f"{ae.value.value}{ae.value.unit}"
            elif is_phone(ae):     val = f"{ae.value.e164}"
            else:                  val = ae.text
            unsupported.append(f"{ae.type}:{val}")

    overall = sum(covered_by_type.values()) / sum(total_by_type.values())
    by_type = {k: covered_by_type.get(k, 0) / v for k, v in total_by_type.items()}
    return {
        "match": {
            "overall": round(overall, 4),
            "by_type": {k: round(v, 4) for k, v in by_type.items()},
            "unsupported": unsupported
        },
        "supported_entities": {
            "items": supported_items,
            "by_type": dict(Counter([it["type"] for it in supported_items])),
            "count": len(supported_items)
        }
    }

# -------------------------
# TF-IDF / embeddings
# -------------------------

def _tfidf_vector(terms: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(terms)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}

def _cosine(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    if not vec1 or not vec2: return 0.0
    dot = sum(v * vec2.get(k, 0.0) for k, v in vec1.items())
    n1 = sqrt(sum(v * v for v in vec1.values()))
    n2 = sqrt(sum(v * v for v in vec2.values()))
    if n1 == 0 or n2 == 0: return 0.0
    return dot / (n1 * n2)

_EMB = None
def _maybe_load_embedder(path: str = "models/all-MiniLM-L6-v2"):
    global _EMB
    if _EMB is not None:
        return _EMB
    try:
        from sentence_transformers import SentenceTransformer
        import os
        if os.path.exists(path):
            _EMB = SentenceTransformer(path, device="cpu")
        else:
            _EMB = None
    except Exception:
        _EMB = None
    return _EMB



def _embed_alignment(question: str, answer: str, q_terms: List[str], idf: Dict[str, float],
                     term_threshold: float = 0.5,
                     model_name: str = "models/all-MiniLM-L6-v2") -> Dict[str, Optional[float]]:
    model = _maybe_load_embedder(model_name)
    if model is None or not question or not answer:
        return {"cosine_embed": None, "answer_covers_question_sem": None}
    from sentence_transformers import util as st_util
    qa_emb = model.encode([question, answer], convert_to_tensor=True, normalize_embeddings=True)
    cos = float(st_util.cos_sim(qa_emb[0], qa_emb[1]).item())
    if not q_terms:
        return {"cosine_embed": round(cos, 4), "answer_covers_question_sem": 1.0}
    term_embs = model.encode(q_terms, convert_to_tensor=True, normalize_embeddings=True)
    sims = st_util.cos_sim(term_embs, qa_emb[1]).squeeze(1)
    total = sum(idf.get(t, 1.0) for t in q_terms) or 1.0
    covered = 0.0
    for term, sim in zip(q_terms, sims):
        if float(sim) >= term_threshold:
            covered += idf.get(term, 1.0)
    return {"cosine_embed": round(cos, 4), "answer_covers_question_sem": round(covered / total, 4)}

def _embed_context_alignment(answer: str, contexts: List[str], 
                            model_name: str = "models/all-MiniLM-L6-v2") -> Dict[str, Optional[float]]:
    model = _maybe_load_embedder(model_name)
    if model is None or not answer or not contexts:
        return {"answer_context_similarity": None, "best_context_similarity": None}
    from sentence_transformers import util as st_util
    all_texts = [answer] + contexts
    embeddings = model.encode(all_texts, convert_to_tensor=True, normalize_embeddings=True)
    answer_emb = embeddings[0]
    context_embs = embeddings[1:]
    similarities = []
    for ctx_emb in context_embs:
        sim = float(st_util.cos_sim(answer_emb, ctx_emb).item())
        similarities.append(sim)
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    best_similarity = max(similarities) if similarities else 0.0
    return {
        "answer_context_similarity": round(avg_similarity, 4),
        "best_context_similarity": round(best_similarity, 4)
    }

# -------------------------
# Unsupported terms helpers
# -------------------------

def _unsupported_terms(ans_terms: List[str], ctx_terms_all: List[str], idf: Dict[str, float]) -> List[Dict[str, float]]:
    a_ctr = Counter(ans_terms)
    c_set = set(ctx_terms_all)
    items = []
    for t, cnt in a_ctr.items():
        if t not in c_set:
            items.append({"term": t, "count": cnt, "idf": idf.get(t, 1.0), "impact": cnt * idf.get(t, 1.0)})
    items.sort(key=lambda x: (-x["impact"], x["term"]))
    for it in items:
        it["idf"] = round(it["idf"], 4)
        it["impact"] = round(it["impact"], 4)
    return items

def _unsupported_terms_per_sentence(answer: str, ctx_terms_all: List[str], idf: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for sent in _split_sentences(answer):
        s_terms = _informative_terms(_tokens(sent))
        items = _unsupported_terms(s_terms, ctx_terms_all, idf)
        out.append({"sentence": sent, "unsupported_terms": items})
    return out

# -------------------------
# Main advanced function
# -------------------------

def context_utilization_report_with_entities(
    question: str,
    answer: str,
    retrieved_contexts: List[str],
    use_bm25_for_best: bool = True,
    use_embed_alignment: bool = False,    # set True if sentence-transformers installed
    embed_term_threshold: float = 0.5,
    extractor_config: Optional[ExtractorConfig] = None
) -> Dict[str, Any]:
    """
    Advanced context utilization analysis with typed entity extraction and sentence similarity.
    QUANTITY extraction is regex+unit-table (no sklearn/quantulum3).
    """
    # Build extractor config (or use default)
    ex_cfg = extractor_config or ExtractorConfig(
        enable_money=True,
        enable_date=True,
        enable_quantity=True,   # safe: now regex+table
        enable_phone=False,
        enable_number=True,
        enable_percent=True,
        timezone="UTC",         # change if you prefer different timezone
    )

    # Edge: no answer
    if not answer or not answer.strip():
        return {
            "precision_token": None,
            "recall_context": None,
            "numeric_match": None,
            "entity_match": {"overall": None, "by_type": {}, "unsupported": []},
            "supported_entities": {"items": [], "by_type": {}, "count": 0},
            "supported_terms": [],
            "supported_terms_per_sentence": [],
            "per_sentence": [],
            "qr_alignment": {
                "cosine_tfidf": None, "answer_covers_question": None,
                "cosine_embed": None, "answer_covers_question_sem": None
            },
            "context_alignment": {
                "answer_context_similarity": None,
                "best_context_similarity": None
            },
            "unsupported_terms": [],
            "unsupported_terms_per_sentence": [],
            "unsupported_numbers": [],
            "summary": "N/A (No answer generated)"
        }

    # Tokenize
    q_tokens = _tokens(question or "")
    a_tokens = _tokens(answer or "")
    ctx_tokens_list = [_tokens(s or "") for s in (retrieved_contexts or [])]

    q_terms = _informative_terms(q_tokens)
    a_terms = _informative_terms(a_tokens)
    ctx_terms_list = [_informative_terms(t) for t in ctx_tokens_list]
    all_ctx_terms = [t for lst in ctx_terms_list for t in lst]

    # Build IDF over contexts + Q + A
    idf = _build_idf((ctx_terms_list or []) + [q_terms, a_terms])

    # If no contexts, still compute alignment and unsupported vs empty
    if not retrieved_contexts:
        vq, va = _tfidf_vector(q_terms, idf), _tfidf_vector(a_terms, idf)
        qr_cos = _cosine(vq, va)
        qr_cov = _weighted_recall(Counter(a_terms), Counter(q_terms), idf)
        embed_align = {"cosine_embed": None, "answer_covers_question_sem": None}
        if use_embed_alignment:
            embed_align = _embed_alignment(question, answer, q_terms, idf, embed_term_threshold)
        return {
            "precision_token": None,
            "recall_context": None,
            "numeric_match": _numeric_match_only(answer, [], config=ex_cfg),
            "entity_match": {"overall": None, "by_type": {}, "unsupported": []},
            "supported_entities": {"items": [], "by_type": {}, "count": 0},
            "supported_terms": [],
            "supported_terms_per_sentence": [],
            "per_sentence": [0.0 for _ in _split_sentences(answer)],
            "qr_alignment": {
                "cosine_tfidf": round(qr_cos, 4),
                "answer_covers_question": round(qr_cov, 4),
                "cosine_embed": embed_align["cosine_embed"],
                "answer_covers_question_sem": embed_align["answer_covers_question_sem"],
            },
            "context_alignment": {
                "answer_context_similarity": None,
                "best_context_similarity": None
            },
            "unsupported_terms": _unsupported_terms(a_terms, [], idf),
            "unsupported_terms_per_sentence": _unsupported_terms_per_sentence(answer, [], idf),
            "unsupported_numbers": _unsupported_numbers(answer, [], config=ex_cfg),
            "summary": "N/A (No retrieved context provided)."
        }

    # ---- Grounding vs contexts
    precision_token = _weighted_precision(a_terms, all_ctx_terms, idf)

    # ---- Best-context recall
    if use_bm25_for_best:
        best_i = _pick_best_context_by_bm25(a_terms, ctx_terms_list, idf)
    else:
        best_i, best_sc = 0, float("-inf")
        for i, terms in enumerate(ctx_terms_list):
            sc = _weighted_precision(terms, a_terms, idf)
            if sc > best_sc:
                best_sc, best_i = sc, i
    best_ctx_terms = ctx_terms_list[best_i] if ctx_terms_list else []
    recall_context = _weighted_recall(Counter(a_terms), Counter(best_ctx_terms), idf)

    # ---- Numeric & Entity alignment (typed)
    numeric_match = _numeric_match_only(answer, retrieved_contexts, config=ex_cfg)
    ent = _entity_alignment(answer, retrieved_contexts, config=ex_cfg)
    entity_match = ent["match"]
    supported_entities = ent["supported_entities"]

    # ---- Per-sentence precision (vs ALL context)
    per_sentence = []
    for sent in _split_sentences(answer):
        s_terms = _informative_terms(_tokens(sent))
        p = _weighted_precision(s_terms, all_ctx_terms, idf) if s_terms else 0.0
        per_sentence.append(round(p, 4))

    # ---- Q↔A alignment
    vq, va = _tfidf_vector(q_terms, idf), _tfidf_vector(a_terms, idf)
    qr_cosine = _cosine(vq, va)
    qr_answer_coverage = _weighted_recall(Counter(a_terms), Counter(q_terms), idf)
    embed_align = {"cosine_embed": None, "answer_covers_question_sem": None}
    if use_embed_alignment:
        embed_align = _embed_alignment(question, answer, q_terms, idf, embed_term_threshold)
    
    # ---- Answer↔Context alignment (semantic similarity)
    context_align = {"answer_context_similarity": None, "best_context_similarity": None}
    if use_embed_alignment:
        context_align = _embed_context_alignment(answer, retrieved_contexts)

    # ---- Supported terms (for UI highlighting)
    supported_terms, supported_terms_per_sentence = _collect_supported_terms(answer, all_ctx_terms, idf)

    # ---- Unsupported (lexical & numeric)
    unsupported = _unsupported_terms(a_terms, all_ctx_terms, idf)
    unsupported_ps = _unsupported_terms_per_sentence(answer, all_ctx_terms, idf)
    unsupported_nums = _unsupported_numbers(answer, retrieved_contexts, config=ex_cfg)

    # ---- Summary
    pct = round(precision_token * 100, 1)
    rec = round(recall_context * 100, 1)
    nump = round(numeric_match * 100, 1)
    entp = round((entity_match["overall"] if entity_match["overall"] is not None else 0.0) * 100, 1)
    parts = [f"{pct}% grounded", f"{rec}% best-context recall", f"{nump}% numeric", f"{entp}% entity"]
    if use_embed_alignment and embed_align["cosine_embed"] is not None:
        parts.append(f"Q↔A embed {round(embed_align['cosine_embed'], 2)}")
        if context_align["answer_context_similarity"] is not None:
            parts.append(f"A↔C embed {round(context_align['answer_context_similarity'], 2)}")
    else:
        parts.append(f"Q↔A tfidf {round(qr_cosine, 2)}")
    summary = "; ".join(parts) + "."

    return {
        "precision_token": round(precision_token, 4),
        "recall_context": round(recall_context, 4),
        "numeric_match": round(numeric_match, 4),
        "entity_match": entity_match,
        "supported_entities": supported_entities,
        "supported_terms": supported_terms,
        "supported_terms_per_sentence": supported_terms_per_sentence,
        "per_sentence": per_sentence,
        "qr_alignment": {
            "cosine_tfidf": round(qr_cosine, 4),
            "answer_covers_question": round(qr_answer_coverage, 4),
            "cosine_embed": embed_align["cosine_embed"],
            "answer_covers_question_sem": embed_align["answer_covers_question_sem"],
        },
        "context_alignment": {
            "answer_context_similarity": context_align["answer_context_similarity"],
            "best_context_similarity": context_align["best_context_similarity"],
        },
        "unsupported_terms": unsupported,
        "unsupported_terms_per_sentence": unsupported_ps,
        "unsupported_numbers": unsupported_nums,
            "summary": summary
    }

# wrapper for context_utilization_report_with_entities
def calculate_context_utilization_percentage(answer: str, context_snippets: List[str]) -> Dict[str, Any]:
    return context_utilization_report_with_entities(
        question="",
        answer=answer,
        retrieved_contexts=context_snippets,
        use_bm25_for_best=True,
        use_embed_alignment=False,
        extractor_config=None,  # or pass a custom ExtractorConfig
    )

# -------------------------
# Simple heuristics
# -------------------------

def calculate_confidence_score(answer: str, context: str, answer_type: str) -> str:
    if not answer or answer.strip() == "":
        return "Low"
    if answer_type == "abstain":
        return "Low"
    answer_length = len(answer.split())
    context_length = len(context.split())
    if answer_length > 10 and context_length > 50:
        return "High"
    elif answer_length > 5 and context_length > 20:
        return "Medium"
    else:
        return "Low"

def calculate_faithfulness_score(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0
    answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    if not answer_words or not context_words:
        return 0.0
    overlap = len(answer_words.intersection(context_words))
    return min(overlap / len(answer_words), 1.0)

def calculate_completeness_score(answer: str, question: str) -> float:
    if not answer or not question:
        return 0.0
    answer_length = len(answer.split())
    question_length = len(question.split())
    if question_length < 5:
        return min(answer_length / 20, 1.0)
    else:
        return min(answer_length / 30, 1.0)