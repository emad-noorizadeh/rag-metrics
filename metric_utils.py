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
import warnings
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from math import log, sqrt

# --- Optional spaCy (POS) ---
_SPACY_NLP = None
def _maybe_load_spacy(model: str = "en_core_web_sm"):
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP
    try:
        import spacy
        _SPACY_NLP = spacy.load(model, disable=["ner"])  # fast POS/lemmatizer only
    except Exception:
        _SPACY_NLP = None
    return _SPACY_NLP

# NEW imports from typed extractor
from extractor import (
    ExtractorConfig, DEFAULT_CONFIG,
    Entity, EntityType,
    extract_entities, extract_by_type,
    is_money, is_number, is_percent, is_date, is_quantity, is_phone,
    money_equal, number_equal, date_equal, entity_coverage,
    canonicalize_date, match_entity_values,   # <-- ensure this is included
)

warnings.filterwarnings(
    "ignore",
    message=".*split_arg_string is deprecated.*",
    category=DeprecationWarning,
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

# Content POS for “semantic” terms (tune to taste)
_CONTENT_POS = {"NOUN", "PROPN", "NUM", "VERB", "ADJ"}

def _spacy_pos_terms(text: str, nlp, stopwords=_STOPWORDS) -> List[str]:
    """Return normalized lemmas for content POS only."""
    if not text:
        return []
    doc = nlp(text)
    out = []
    for t in doc:
        if t.pos_ in _CONTENT_POS:
            lem = (t.lemma_ or t.text).lower().strip()
            if lem and lem.isascii() and lem not in stopwords:
                # normalize simple digits the same way we do elsewhere
                if re.fullmatch(r"[-+]?\d[\d,]*\.?\d*", lem):
                    lem = re.sub(r"[,\s]", "", lem)
                out.append(lem)
    return out

def _head_nouns(text: str, nlp) -> List[str]:
    """Extract head nouns of noun chunks for stronger concept alignment."""
    if not text:
        return []
    doc = nlp(text)
    heads = []
    for nc in doc.noun_chunks:
        h = nc.root.lemma_.lower() if nc.root.lemma_ else nc.root.text.lower()
        if h and h.isascii():
            heads.append(h)
    return heads

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
# POS-aware unsupported/supported calculations
# -------------------------

def _pos_supported_stats(answer: str, contexts: List[str], idf: Dict[str, float], nlp) -> Dict[str, float]:
    """
    Compute POS-aware coverage:
      - content_precision_token: IDF-weighted precision using only content-POS terms
      - content_unsupported_mass: 1 - precision over content terms
      - content_term_support_rate: fraction of unique content terms supported
      - head_noun_support_rate: fraction of answer head nouns found in context
    Returns zeros if spaCy not available or texts missing.
    """
    if nlp is None or not answer:
        return {
            "content_precision_token": None,
            "content_unsupported_mass": None,
            "content_term_support_rate": None,
            "head_noun_support_rate": None,
        }

    a_pos_terms = _spacy_pos_terms(answer, nlp)
    if not contexts:
        # no context—define “none supported”
        uniq = len(set(a_pos_terms)) or 1
        return {
            "content_precision_token": 0.0 if a_pos_terms else None,
            "content_unsupported_mass": 1.0 if a_pos_terms else None,
            "content_term_support_rate": 0.0 if a_pos_terms else None,
            "head_noun_support_rate": 0.0 if _head_nouns(answer, nlp) else None,
        }

    ctx_pos_terms = []
    for s in contexts:
        ctx_pos_terms.extend(_spacy_pos_terms(s or "", nlp))

    # IDF precision on content terms
    c_prec = _weighted_precision(a_pos_terms, ctx_pos_terms, idf) if a_pos_terms else None
    c_unsup_mass = (1.0 - c_prec) if (c_prec is not None) else None

    # Unique-term support rate
    a_set = set(a_pos_terms)
    c_set = set(ctx_pos_terms)
    support_rate = (len(a_set & c_set) / len(a_set)) if a_set else None

    # Head noun support rate (unique heads overlapped)
    a_heads = set(_head_nouns(answer, nlp))
    ctx_heads = set()
    for s in contexts:
        ctx_heads.update(_head_nouns(s or "", nlp))
    head_rate = (len(a_heads & ctx_heads) / len(a_heads)) if a_heads else None

    # Round lightly for report neatness
    def r(x): return None if x is None else round(float(x), 4)
    return {
        "content_precision_token": r(c_prec),
        "content_unsupported_mass": r(c_unsup_mass),
        "content_term_support_rate": r(support_rate),
        "head_noun_support_rate": r(head_rate),
    }

# -------------------------
# Numeric coverage (typed)
# -------------------------

def _numeric_match_only(answer: str, contexts: List[str], config: ExtractorConfig = DEFAULT_CONFIG) -> float:
    """
    Fraction of numeric facts (MONEY/NUMBER/PERCENT) in answer present in any context.
    Uses typed extractor (no spaCy/quantulum3) with overlap suppression so that
    higher-priority entities (MONEY > DATE > QUANTITY > PHONE > PERCENT > NUMBER)
    suppress lower-priority overlaps before matching.
    """
    target_types = {"MONEY", "NUMBER", "PERCENT"}

    # Extract with suppression to avoid NUMBERs that are part of DATE/MONEY, etc.
    ans_all = _extract_with_suppression(answer or "", config)
    ans = [e for e in ans_all if e.type in target_types]
    if not ans:
        return 1.0

    ctx_all: List[Entity] = []
    for s in contexts or []:
        ctx_all.extend(_extract_with_suppression(s or "", config))
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

    # Extract with suppression (avoid NUMBER inside DATE/MONEY etc.)
    ans_all = _extract_with_suppression(answer or "", config)
    ctx_all: List[Entity] = []
    for s in contexts or []:
        ctx_all.extend(_extract_with_suppression(s or "", config))

    ans = [e for e in ans_all if e.type in target_types]
    ctx = [e for e in ctx_all if e.type in target_types]

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

def _suppress_overlaps_by_priority(ents: List[Entity]) -> List[Entity]:
    """
    Suppress lower-priority entities that overlap higher-priority ones.

    Priority (highest → lowest):
      1) MONEY
      2) DATE
      3) QUANTITY
      4) PHONE
      5) PERCENT
      6) NUMBER

    If two entities overlap, keep the higher-priority one and drop the lower.
    Equal-priority overlaps are both kept.
    """
    if not ents:
        return ents

    prio_order = ["MONEY", "DATE", "QUANTITY", "PHONE", "PERCENT", "NUMBER"]
    prio = {t: i for i, t in enumerate(prio_order)}

    # Sort by priority (high first), then by longer span first, then by start
    def keyfn(e: Entity):
        length = (e.span[1] - e.span[0]) if e.span else 0
        return (prio.get(e.type, len(prio_order)), -length, e.span[0])

    sorted_ents = sorted(ents, key=keyfn)
    kept: List[Tuple[int, Tuple[int, int], Entity]] = []

    for e in sorted_ents:
        p_cur = prio.get(e.type, len(prio_order))
        s, t = e.span
        overlap_with_higher = False
        for p_k, (ks, kt), _ in kept:
            if p_k < p_cur:  # only suppress if an already-kept entity has higher priority
                if not (t <= ks or kt <= s):  # spans overlap
                    overlap_with_higher = True
                    break
        if not overlap_with_higher:
            kept.append((p_cur, (s, t), e))

    # Return kept entities in original text order for downstream stability
    return [e for _, (_, _), e in sorted(kept, key=lambda x: (x[1][0], x[1][1]))]


def _extract_with_suppression(text: str, config: ExtractorConfig) -> List[Entity]:
    """Extract entities then suppress lower-priority overlaps."""
    ents = extract_entities(text or "", config=config)
    return _suppress_overlaps_by_priority(ents)


def _entity_alignment(answer: str, contexts: List[str], config: ExtractorConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """
    Typed entity alignment using extractor.py (regex money/number/percent/date/quantity/phone).
    Returns match (overall/by_type/unsupported) and supported_entities with spans.
    """
    ans_ents: List[Entity] = _extract_with_suppression(answer or "", config)
    ctx_ents: List[Entity] = []
    for s in contexts or []:
        ctx_ents.extend(_extract_with_suppression(s or "", config))

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
    # Tri-state entity presence/match per type
    all_types: List[EntityType] = ["MONEY","NUMBER","PERCENT","DATE","QUANTITY","PHONE"]
    presence_by_type = {t: int(total_by_type.get(t, 0) > 0) for t in all_types}
    state_by_type: Dict[str, float] = {}
    all_matched_by_type: Dict[str, int] = {}
    for t in all_types:
        tot = total_by_type.get(t, 0)
        if tot == 0:
            state = 0.0
            all_matched = 0
        else:
            fully_covered = covered_by_type.get(t, 0) == tot
            state = 1.0 if fully_covered else 0.0
            all_matched = 1 if fully_covered else 0
        state_by_type[t] = state
        all_matched_by_type[t] = all_matched
    return {
        "match": {
            "overall": round(overall, 4),
            # removed: per-type coverage in favor of tri-state
            "unsupported": unsupported,
            "presence_by_type": presence_by_type,
            "state_by_type": state_by_type,
            "all_matched_by_type": all_matched_by_type,
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
        candidates = []

        # Absolute paths are tried as-is.
        if path and os.path.isabs(path):
            candidates.append(path)
        else:
            # Path relative to current working directory (legacy behaviour)
            if path:
                candidates.append(path)

            # Path relative to env override
            env_dir = os.environ.get("RAG_MODELS_DIR") or os.environ.get("MODELS_DIR")
            if env_dir and path:
                candidates.append(os.path.join(env_dir, path))

            # Path relative to repo root (../models/…)
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if path:
                candidates.append(os.path.normpath(os.path.join(repo_root, path)))

        repo_parent = os.path.dirname(repo_root)
        if path:
            candidates.append(os.path.normpath(os.path.join(repo_parent, path)))

        for cand in candidates:
            if cand and os.path.exists(cand):
                _EMB = SentenceTransformer(cand, device="cpu")
                break
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


def _compute_unsupported_extras(
    answer_terms: List[str],
    unsupported: List[Dict[str, Any]],
    precision_token: Optional[float],
    numeric_match: Optional[float],
    entity_match: Dict[str, Any],
    topk: int = 5,
) -> Dict[str, Any]:
    """Compute the six requested unsupported metrics.

    - unsupported_mass = 1 - precision_token
    - unsupported_topk_impact = sum(idf*count) for top-K unsupported terms
    - unsupported_term_rate = |{unsupported unique terms}| / |{unique answer terms}|
    - hallucination_risk_score = unsupported_mass (without POS filtering)
    - unsupported_numeric_rate = 1 - numeric_match
    - unsupported_entity_mass = 1 - entity_match.overall
    """
    # Unsupported Mass (%): relies directly on precision_token
    if precision_token is None:
        unsupported_mass = None
    else:
        unsupported_mass = max(0.0, min(1.0, 1.0 - float(precision_token)))

    # Unsupported Top-K Impact
    topk_terms = unsupported[: max(0, int(topk))]
    topk_impact = sum(u.get("impact", 0.0) for u in topk_terms)

    # Unsupported Term Rate
    uniq_answer_terms = set(answer_terms)
    uniq_unsupported_terms = {u.get("term") for u in unsupported}
    unsupported_term_rate = (
        (len(uniq_unsupported_terms) / max(1, len(uniq_answer_terms))) if uniq_answer_terms else 0.0
    )

    # Hallucination Risk Score (no POS filter -> equals unsupported_mass)
    hallucination_risk_score = unsupported_mass

    # Unsupported-Numeric Rate
    unsupported_numeric_rate = None if numeric_match is None else max(0.0, min(1.0, 1.0 - float(numeric_match)))

    # Unsupported-Entity Mass
    overall_ent = None if not entity_match else entity_match.get("overall")
    unsupported_entity_mass = None if overall_ent is None else max(0.0, min(1.0, 1.0 - float(overall_ent)))

    return {
        "unsupported_mass": None if unsupported_mass is None else round(unsupported_mass, 4),
        "unsupported_topk_impact": round(float(topk_impact), 4),
        "unsupported_topk_terms": topk_terms,
        "unsupported_term_rate": round(float(unsupported_term_rate), 4),
        "hallucination_risk_score": None if hallucination_risk_score is None else round(float(hallucination_risk_score), 4),
        "unsupported_numeric_rate": None if unsupported_numeric_rate is None else round(float(unsupported_numeric_rate), 4),
        "unsupported_entity_mass": None if unsupported_entity_mass is None else round(float(unsupported_entity_mass), 4),
    }


# -------------------------
# Inference detector (paraphrase vs inference)
# -------------------------
def _inference_signal(rep: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic detector for 'likely inferred (not paraphrased)' answers:
      - low lexical overlap, but
      - high numeric/entity coverage, and
      - moderate Q↔A alignment.
    Returns a score in [0,1] plus a boolean flag.
    """
    prec = rep.get("precision_token")
    cprec = rep.get("content_precision_token")
    num = rep.get("numeric_match")
    ent_overall = (rep.get("entity_match", {}) or {}).get("overall")
    qa_tfidf = (rep.get("qr_alignment", {}) or {}).get("cosine_tfidf")

    # defaults
    prec = 0.0 if prec is None else float(prec)
    cprec = cprec if (cprec is not None) else prec
    num = 0.0 if num is None else float(num)
    ent_overall = 0.0 if ent_overall is None else float(ent_overall)
    qa_tfidf = 0.0 if qa_tfidf is None else float(qa_tfidf)

    # Conditions (tune thresholds as you like)
    low_lex = (prec < 0.45) or (cprec < 0.45)
    strong_typed = max(num, ent_overall) >= 0.9
    sem_ok = qa_tfidf >= 0.35  # embeddings could replace this if enabled

    # Score: how strongly it looks inferred
    score = 0.0
    if low_lex and strong_typed and sem_ok:
        # weight by how low lexical is and how high typed grounding is
        lex_gap = 1.0 - max(prec, (cprec if cprec is not None else 0.0))
        typed = max(num, ent_overall)
        score = min(1.0, 0.5 * lex_gap + 0.5 * (typed - 0.9) * 10.0)  # cheap squeeze

    return {
        "inference_likely": bool(score >= 0.5),
        "inference_score": round(score, 4),
        "inference_explanation": (
            "Low lexical grounding but strong numeric/entity alignment and decent Q↔A similarity."
            if score >= 0.5 else
            "No strong inference signal detected."
        )
    }


# -------------------------
# Main advanced function
# -------------------------

def context_utilization_report_with_entities(
    question: str,
    answer: str,
    retrieved_contexts: List[str],
    use_bm25_for_best: bool = True,
    use_embed_alignment: bool = True,    # set True if sentence-transformers installed
    embed_term_threshold: float = 0.5,
    extractor_config: Optional[ExtractorConfig] = None,
    timezone: str = "UTC",
    metrics_config: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Advanced context utilization analysis with typed entity extraction and sentence similarity.
    QUANTITY extraction is regex+unit-table (no sklearn/quantulum3).
    """

    def _is_enabled(cfg: Optional[Dict[str, bool]], name: str, default: bool = True) -> bool:
        if cfg is None:
            return default
        # Accept dict-like and SimpleNamespace-like configs
        try:
            if isinstance(cfg, dict):
                return bool(cfg.get(name, default))
            # fall back to getattr for SimpleNamespace or similar objects
            return bool(getattr(cfg, name, default))
        except Exception:
            return default
    # Build extractor config (or use default). We now enable spaCy-fusion by default
    # so you benefit from the fusion scorer/overlap-suppression automatically when
    # spaCy is installed; extractor will silently no-op if spaCy is missing.
    ex_cfg = (
        extractor_config
        or (getattr(metrics_config, "extractor", None) if metrics_config is not None else None)
        or ExtractorConfig(
            enable_money=True,
            enable_date=True,
            enable_quantity=True,   # safe: regex+unit-table
            enable_phone=False,
            enable_number=True,
            enable_percent=True,
            timezone=timezone,
            # NEW: fusion knobs (ExtractorConfig supports these; harmless if absent)
            use_spacy_fusion=True,
            prefer_deterministic=True,
        )
    )
    # Allow a simple switch via metrics_config without requiring callers to
    # construct an ExtractorConfig explicitly.
    if metrics_config is not None:
        mc = metrics_config

        def _mc_get(name: str):
            if isinstance(mc, dict):
                return mc.get(name, None)
            return getattr(mc, name, None)

        # Optional overrides if provided directly at the top level
        _val = _mc_get("use_spacy_fusion")
        if _val is not None:
            try:
                setattr(ex_cfg, "use_spacy_fusion", bool(_val))
            except Exception:
                pass

        _val = _mc_get("prefer_deterministic")
        if _val is not None:
            try:
                setattr(ex_cfg, "prefer_deterministic", bool(_val))
            except Exception:
                pass

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

    # Per-sentence summary features (quick-win)
    min_sentence_precision = min(per_sentence) if per_sentence else 0.0
    mean_sentence_precision = (sum(per_sentence) / len(per_sentence)) if per_sentence else 0.0
    # p90
    if per_sentence:
        sorted_ps = sorted(per_sentence)
        idx90 = int(0.9 * (len(sorted_ps) - 1))
        p90_sentence_precision = sorted_ps[idx90]
    else:
        p90_sentence_precision = 0.0

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
    extras = _compute_unsupported_extras(
        a_terms,
        unsupported,
        precision_token,
        numeric_match,
        entity_match,
        topk=5,
    )

    # ---- Quick-win features (conditionally included)
    quickwin = {}
    # 1) Unsupported footprint size
    if _is_enabled(metrics_config, "unsupported_term_count"):
        quickwin["unsupported_term_count"] = float(len(unsupported))
    if _is_enabled(metrics_config, "unsupported_topk_count"):
        quickwin["unsupported_topk_count"] = float(len(extras.get("unsupported_topk_terms", []) or []))
    # 2) Supported vs unsupported mass balance (use same top-k K as in extras)
    if _is_enabled(metrics_config, "supported_topk_impact") or _is_enabled(metrics_config, "unsupported_to_supported_ratio"):
        K = len(extras.get("unsupported_topk_terms", []) or [])
        if K > 0:
            # supported_terms is a list of {term,count,idf}
            sup_sorted = sorted(supported_terms, key=lambda x: -(x.get("idf", 1.0) * x.get("count", 1)))
            supported_topk_impact = sum((it.get("idf", 1.0) * it.get("count", 1)) for it in sup_sorted[:K])
        else:
            supported_topk_impact = 0.0
        if _is_enabled(metrics_config, "supported_topk_impact"):
            quickwin["supported_topk_impact"] = round(float(supported_topk_impact), 4)
        if _is_enabled(metrics_config, "unsupported_to_supported_ratio"):
            uns_topk = float(extras.get("unsupported_topk_impact", 0.0) or 0.0)
            quickwin["unsupported_to_supported_ratio"] = round(uns_topk / (supported_topk_impact + 1e-6), 4)
    # 3) Numeric/entity mismatch counts
    if _is_enabled(metrics_config, "unsupported_numeric_count"):
        quickwin["unsupported_numeric_count"] = float(len(unsupported_nums))
    if _is_enabled(metrics_config, "unsupported_entity_count"):
        quickwin["unsupported_entity_count"] = float(len(entity_match.get("unsupported", []) or []))
    # 4) Best-context alignment proxy and volume
    if _is_enabled(metrics_config, "best_context_len"):
        quickwin["best_context_len"] = float(len(best_ctx_terms))
    if _is_enabled(metrics_config, "best_context_share"):
        quickwin["best_context_share"] = round(float(precision_token * recall_context), 4)
    if _is_enabled(metrics_config, "num_contexts"):
        quickwin["num_contexts"] = float(len(retrieved_contexts or []))
    if _is_enabled(metrics_config, "avg_ctx_len_tokens"):
        avg_ctx_len_tokens = 0.0
        if ctx_tokens_list:
            total = sum(len(toks) for toks in ctx_tokens_list)
            avg_ctx_len_tokens = total / len(ctx_tokens_list)
        quickwin["avg_ctx_len_tokens"] = round(float(avg_ctx_len_tokens), 4)
    # 5) QA lexical alignment extras
    if _is_enabled(metrics_config, "q_len"):
        quickwin["q_len"] = float(len(q_terms))
    if _is_enabled(metrics_config, "a_len"):
        quickwin["a_len"] = float(len(a_terms))
    if _is_enabled(metrics_config, "qa_len_ratio"):
        quickwin["qa_len_ratio"] = round(float(len(a_terms) / max(1, len(q_terms))), 4)
    # 7) Per-sentence summary stats
    if _is_enabled(metrics_config, "min_sentence_precision"):
        quickwin["min_sentence_precision"] = round(float(min_sentence_precision), 4)
    if _is_enabled(metrics_config, "mean_sentence_precision"):
        quickwin["mean_sentence_precision"] = round(float(mean_sentence_precision), 4)
    if _is_enabled(metrics_config, "p90_sentence_precision"):
        quickwin["p90_sentence_precision"] = round(float(p90_sentence_precision), 4)

    # ---- POS-aware stats (optional spaCy)
    pos_stats = {}
    if _is_enabled(metrics_config, "enable_pos_metrics", True):
        nlp = _maybe_load_spacy("en_core_web_sm")
        pos_stats = _pos_supported_stats(answer, retrieved_contexts, idf, nlp)
    else:
        pos_stats = {
            "content_precision_token": None,
            "content_unsupported_mass": None,
            "content_term_support_rate": None,
            "head_noun_support_rate": None,
        }

    # ---- Inference signal
    # Merge pos_stats into a temp dict so _inference_signal can see content_precision_token
    tmp_rep_for_infer = {
        **{
            "precision_token": precision_token,
            "numeric_match": numeric_match,
            "entity_match": entity_match,
            "qr_alignment": {"cosine_tfidf": qr_cosine},
        },
        **pos_stats,
    }
    infer = {}
    if _is_enabled(metrics_config, "enable_inference_signal", True):
        infer = _inference_signal(tmp_rep_for_infer)
    else:
        infer = {"inference_likely": None, "inference_score": None, "inference_explanation": None}

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
        # POS-aware additions
        "content_precision_token": pos_stats.get("content_precision_token"),
        "content_unsupported_mass": pos_stats.get("content_unsupported_mass"),
        "content_term_support_rate": pos_stats.get("content_term_support_rate"),
        "head_noun_support_rate": pos_stats.get("head_noun_support_rate"),
        # Inference detector
        "inference_likely": infer.get("inference_likely"),
        "inference_score": infer.get("inference_score"),
        "inference_explanation": infer.get("inference_explanation"),
        "summary": summary,
        **quickwin,
        **extras,
    }

# wrapper for context_utilization_report_with_entities
def calculate_context_utilization_percentage(
    answer: str,
    context_snippets: List[str],
    metrics_config: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    return context_utilization_report_with_entities(
        question="",
        answer=answer,
        retrieved_contexts=context_snippets,
        use_bm25_for_best=True,
        use_embed_alignment=True,
        extractor_config=None,  # or pass a custom ExtractorConfig
        metrics_config=metrics_config,
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
