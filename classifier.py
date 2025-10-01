# demo_train_classifier.py
from typing import List, Dict, Any, Tuple, Iterable, Optional
from dataclasses import dataclass, field
import re
import numpy as np
from sklearn.linear_model import LogisticRegression

# uses your existing file
from metric_utils import context_utilization_report_with_entities

# -----------------------------
# 1) Tiny labeled toy dataset
# -----------------------------
Dataset = List[Dict[str, Any]]

toy_data: Dataset = [
    # ====== FINANCE ======
    dict(
        q="What was the 2023 revenue and guidance?",
        a="Revenue was $5.2B and guidance is 12% growth.",
        c=[
            "In 2023, the company reported revenue of $5.2B.",
            "Management expects growth of 12 percent next year."
        ],
        y=1,
    ),
    dict(
        q="What was the 2023 revenue?",
        a="Revenue was $5.2B.",
        c=["2023 revenue came in at $4.8B according to the annual report."],
        y=0,  # numeric mismatch
    ),
    dict(
        q="What is Q4 gross margin?",
        a="Q4 gross margin was 45%.",
        c=["Q4 gross margin expanded to 45% on cost optimizations."],
        y=1,
    ),
    dict(
        q="What is Q4 gross margin?",
        a="Q4 gross margin was 47%.",
        c=["Q4 gross margin expanded to 45% on cost optimizations."],
        y=0,  # mismatch
    ),

    # ====== DATES & ENTITIES ======
    dict(
        q="When did the event take place?",
        a="It took place on Jan 3, 2024.",
        c=["The kickoff event occurred on Jan 3, 2024 at 10:30 local time."],
        y=1,
    ),
    dict(
        q="When did the event take place?",
        a="It took place on Feb 2, 2024.",
        c=["The kickoff event occurred on Jan 3, 2024 at 10:30 local time."],
        y=0,  # date mismatch
    ),

    # ====== FACTUAL TEXT (non-numeric) ======
    dict(
        q="Which database engine powers the service?",
        a="The service runs on PostgreSQL.",
        c=["We use PostgreSQL 14 for transactional storage; analytics is on BigQuery."],
        y=1,
    ),
    dict(
        q="Which database engine powers the service?",
        a="The service runs on MongoDB.",
        c=["We use PostgreSQL 14 for transactional storage; analytics is on BigQuery."],
        y=0,
    ),

    # ====== MIXED CONTEXT ======
    dict(
        q="What are the battery and range specs?",
        a="Battery is 500 mAh and range is 10 km.",
        c=["Specs: battery capacity 500 mAh. Typical range is 10 km on eco mode."],
        y=1,
    ),
    dict(
        q="What are the battery and range specs?",
        a="Battery is 700 mAh and range is 12 km.",
        c=["Specs: battery capacity 500 mAh. Typical range is 10 km on eco mode."],
        y=0,  # quantity mismatch both terms
    ),

    # ====== PARTIAL SUPPORT ======
    dict(
        q="Summarize the pricing.",
        a="Plan A is $20, Plan B is $50, and both include phone support.",
        c=["Pricing: Plan A is $20. Plan B is $50.", "Support: email only for both plans."],
        y=0,  # numbers match, but "phone support" is unsupported content
    ),
    dict(
        q="Summarize the pricing.",
        a="Plan A is $20 and Plan B is $50.",
        c=["Pricing: Plan A is $20. Plan B is $50."],
        y=1,
    ),

    # ====== PARAPHRASES / LEXICAL VARIANTS ======
    dict(
        q="What’s the forecast growth?",
        a="They projected twelve percent growth.",
        c=["Management expects growth of 12 percent next year."],
        y=1,
    ),
    dict(
        q="What’s the forecast growth?",
        a="They projected double-digit growth of 25%.",
        c=["Management expects growth of 12 percent next year."],
        y=0,
    ),

    # ====== AMBIGUOUS / UNSUPPORTED NEW CLAIMS ======
    dict(
        q="Which language is the SDK in?",
        a="The SDK is in Python and includes a Rust extension.",
        c=["The SDK is offered in Python."],
        y=0,  # “Rust extension” is unsupported extra claim
    ),
    dict(
        q="Which language is the SDK in?",
        a="The SDK is in Python.",
        c=["The SDK is offered in Python."],
        y=1,
    ),
]


# ----------------------------------------------------------
# 2) Generic feature flattener over the report (no new deps)
# ----------------------------------------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))

def _flatten_report(
    rep: Dict[str, Any],
    prefix: str = "",
    include_list_lengths: bool = True,
) -> Dict[str, float]:
    """
    Recursively flatten the report dict, keeping only numeric scalars.
    For lists: optionally add a '<key>__len' feature.
    For nested dicts: join path with '.' (e.g., 'entity_match.overall').
    """
    out: Dict[str, float] = {}
    for k, v in rep.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if _is_number(v):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_flatten_report(v, prefix=key, include_list_lengths=include_list_lengths))
        elif isinstance(v, list):
            if include_list_lengths:
                out[f"{key}__len"] = float(len(v))
            # ignore list contents for now (keeps it dependency-free & simple)
        # ignore strings/None/etc
    return out


# --------------------------------------------------------
# 3) Feature selection configuration (include/exclude)
# --------------------------------------------------------
@dataclass
class FeatureConfig:
    """
    Controls which flattened features are included for the model.
    - allowlist: exact feature keys to keep (if empty, keep all)
    - denylist: exact feature keys to drop
    - allow_patterns: regex patterns; if set, feature must match at least one
    - deny_patterns: regex patterns to drop
    """
    allowlist: Iterable[str] = field(default_factory=list)
    denylist: Iterable[str] = field(default_factory=list)
    allow_patterns: Iterable[str] = field(default_factory=list)
    deny_patterns: Iterable[str] = field(default_factory=list)

    def filter(self, feats: Dict[str, float]) -> Dict[str, float]:
        keys = list(feats.keys())
        # exact allowlist
        if self.allowlist:
            keys = [k for k in keys if k in self.allowlist]
        # regex allow
        if self.allow_patterns:
            allow_re = [re.compile(p) for p in self.allow_patterns]
            keys = [k for k in keys if any(r.search(k) for r in allow_re)]
        # exact denylist
        if self.denylist:
            keys = [k for k in keys if k not in self.denylist]
        # regex deny
        if self.deny_patterns:
            deny_re = [re.compile(p) for p in self.deny_patterns]
            keys = [k for k in keys if not any(r.search(k) for r in deny_re)]
        return {k: feats[k] for k in keys}


# -----------------------------------------------------------------
# 4) Report wrapper → flattened feature vector (with config toggles)
# -----------------------------------------------------------------
def extract_features_all(
    q: str,
    a: str,
    ctx: List[str],
    metrics_config: Optional[Dict[str, Any]] = None,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    rep = context_utilization_report_with_entities(
        question=q,
        answer=a,
        retrieved_contexts=ctx,
        use_bm25_for_best=True,
        use_embed_alignment=False,       # no embeddings by default (portable)
        metrics_config=metrics_config,   # pass caller’s toggles for quick-wins etc.
    )

    flat = _flatten_report(rep, prefix="")
    if feature_config:
        flat = feature_config.filter(flat)

    # Stable feature order
    names = sorted(flat.keys())
    x = np.array([flat[n] for n in names], dtype=float)
    return x, rep, names


# -------------------------------------------------
# 5) A classifier that uses *all* numeric features
# -------------------------------------------------
@dataclass
class RagFaithfulnessClassifier:
    strict_numeric_gate: bool = True         # hard/soft guard for numeric mismatch
    C: float = 1.0                           # logistic regularization strength
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    metrics_config: Optional[Dict[str, Any]] = None  # forwarded to report

    def __post_init__(self):
        # balanced = robust with small or skewed datasets
        self.model = LogisticRegression(max_iter=300, class_weight="balanced", C=self.C)
        self.feature_names_: List[str] = []

    def featurize_batch(self, rows: Dataset) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        X, y, reps = [], [], []
        feature_names_ref: Optional[List[str]] = None
        for r in rows:
            x, rep, names = extract_features_all(
                r["q"], r["a"], r["c"],
                metrics_config=self.metrics_config,
                feature_config=self.feature_config,
            )
            if feature_names_ref is None:
                feature_names_ref = names
            else:
                # align columns if a later row is missing some keys
                if names != feature_names_ref:
                    # union & re-project everything
                    feature_names_ref = sorted(set(feature_names_ref) | set(names))
                    # re-extract for all accumulated rows to align (cheap for toy; fine for small batches)
                    X = [self._reproject(x_vec, old_names, feature_names_ref) for x_vec, old_names in zip(X, self._names_history)]
                # fallthrough: names matches ref
            X.append(x)
            y.append(int(r["y"]))
            reps.append(rep)
            # remember per-row names for potential re-projection later
            self._names_history = getattr(self, "_names_history", []) + [names]

        # Final projection in case of any misalignment happened mid-way
        if feature_names_ref is None:
            feature_names_ref = []
        X = [self._reproject(x_vec, names, feature_names_ref) for x_vec, names in zip(X, self._names_history)]
        self.feature_names_ = feature_names_ref
        return np.vstack(X) if X else np.zeros((0, 0)), np.array(y, dtype=int), reps

    @staticmethod
    def _reproject(x_vec: np.ndarray, old_names: List[str], new_names: List[str]) -> np.ndarray:
        """Map feature vector from old_names → new_names order, filling missing with 0."""
        old_index = {n: i for i, n in enumerate(old_names)}
        out = np.zeros(len(new_names), dtype=float)
        for i, n in enumerate(new_names):
            j = old_index.get(n, -1)
            if j >= 0:
                out[i] = x_vec[j]
        return out

    def fit(self, rows: Dataset):
        X, y, _ = self.featurize_batch(rows)
        if X.shape[0] == 0:
            raise RuntimeError("No training data.")
        if X.shape[1] == 0:
            raise RuntimeError("No numeric features after filtering.")
        self.model.fit(X, y)
        return self

    def predict_proba(self, q: str, a: str, ctx: List[str]) -> Tuple[float, Dict[str, Any]]:
        x, rep, names = extract_features_all(
            q, a, ctx,
            metrics_config=self.metrics_config,
            feature_config=self.feature_config,
        )
        # align to training feature order
        x = self._reproject(x, names, self.feature_names_)
        p = float(self.model.predict_proba([x])[0, 1])

        # Optional numeric hard/soft gate
        if self.strict_numeric_gate:
            unr = rep.get("unsupported_numeric_rate")
            if isinstance(unr, (int, float)) and unr > 0:
                p = min(p, 0.20)  # clip; or set to 0.0 for hard fail
        return p, rep

    def predict(self, q: str, a: str, ctx: List[str], threshold: float = 0.5) -> int:
        p, _ = self.predict_proba(q, a, ctx)
        return int(p >= threshold)


# -----------------------------
# 6) Quick smoke train & test
# -----------------------------
if __name__ == "__main__":
    # Example: include everything except long per-sentence arrays (we already flatten lengths)
    feature_cfg = FeatureConfig(
        # allowlist=[],  # keep empty to let everything numeric through
        deny_patterns=[
            r"\.by_type\.",                    # drop per-entity-type scatter if not needed
            r"supported_terms(_per_sentence)?",# drop text-detail arrays; we keep lengths via __len
            r"unsupported_terms(_per_sentence)?",
            r"supported_entities\.(items|by_type|count)",  # often noisy; count can be derived
        ],
    )

    # Example: pass quick-win toggles into the report (optional)
    metrics_cfg = {
        # turn specific extras on/off here as you like
        "unsupported_to_supported_ratio": True,
        "supported_topk_impact": True,
        "avg_ctx_len_tokens": True,
        "p90_sentence_precision": True,
        # leave others omitted = default ON (per your metric_utils default behavior)
    }

    # train on first 12, test on last 4 (toy split for demo only)
    train_rows = toy_data[:12]
    test_rows  = toy_data[12:]

    clf = RagFaithfulnessClassifier(
        strict_numeric_gate=True,
        C=1.0,
        feature_config=feature_cfg,
        metrics_config=metrics_cfg,
    ).fit(train_rows)

    print(f"Num features used: {len(clf.feature_names_)}")
    # Uncomment to inspect feature names:
    # for n in clf.feature_names_: print(n)

    print("\n=== Test predictions ===")
    correct = 0
    for i, r in enumerate(test_rows, start=1):
        p, rep = clf.predict_proba(r["q"], r["a"], r["c"])
        yhat = int(p >= 0.5)
        correct += int(yhat == r["y"])
        print(f"[{i}] y={r['y']}  p={p:.3f}  yhat={yhat}  |  q='{r['q']}'")

    acc = correct / len(test_rows)
    print(f"\nToy accuracy: {acc:.2f}  ({correct}/{len(test_rows)})")