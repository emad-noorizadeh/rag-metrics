# scripts/eval_kfold.py
"""
Train/CV on your processed training set, pick best C and threshold, and evaluate on your held-out test set:
python scripts/eval_kfold.py \
  --train-npz processed/train_v1.npz \
  --test-npz  processed/test_v1.npz \
  --Cs 0.1,0.3,1,3,10 \
  --n-splits 5 \
  --objective f1 \
  --min-precision 0.85 \
  --seed 42 \
  --save-model artifacts/lr_model.pkl \
  --save-report artifacts/cv_report.json

  --objective: use f1 (default), or fbeta (with --beta 2.0 for recall-leaning), 
  or fixed_05 if you want a static 0.5 threshold.
	•	--min-precision: enforce a minimum precision when picking thresholds; 
    great if you’d rather avoid false positives.

Notes
	•	This uses plain sklearn LogisticRegression on your precomputed X from data_processing. 
    That’s ideal for CV speed and makes results reproducible.
	•	It prints the top positive/negative coefficients (feature importance proxy) 
    if your NPZ included feature_names.
	•	The saved pickle contains the fitted classifier, the chosen threshold, 
    and feature names so you can load it to score new batches consistently.

# --use-embed-alignment: recorded in artifacts (for reproducibility); features must already include embed metrics from data_processing.
"""
import argparse, json, math, csv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    f1_score, brier_score_loss, precision_score, recall_score
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
import hashlib


# --- Helper: extract inner LogisticRegression (works with Pipeline or plain LR)
def _extract_lr(est):
    if isinstance(est, LogisticRegression):
        return est
    # Try scikit-learn Pipeline-like
    named = getattr(est, "named_steps", None)
    if named:
        # walk from the end to find the last LR
        for _, step in reversed(list(named.items())):
            if isinstance(step, LogisticRegression):
                return step
    # Try generic steps attribute (just in case)
    steps = getattr(est, "steps", None)
    if steps:
        for _, step in reversed(steps):
            if isinstance(step, LogisticRegression):
                return step
    return None


def apply_feature_filters(
    X: np.ndarray,
    feature_names: List[str],
    allow: Optional[List[str]] = None,
    deny: Optional[List[str]] = None,
):
    """Apply simple allow/deny filters to feature matrix and names.

    Returns (X_filtered, filtered_feature_names, mask, meta_dict)
    where mask is a boolean array over the original feature order.
    """
    allow = list(dict.fromkeys(f for f in (allow or []) if f))
    deny = list(dict.fromkeys(f for f in (deny or []) if f))
    if not allow and not deny:
        mask = np.ones(len(feature_names), dtype=bool)
        meta = {
            "allow": [],
            "deny": [],
            "original_feature_count": len(feature_names),
            "kept_feature_count": len(feature_names),
            "dropped_feature_count": 0,
            "dropped_features": [],
            "applied_by": "eval_kfold.py",
        }
        return X, feature_names, mask, meta

    if feature_names is None:
        raise SystemExit("Feature filtering requires feature_names in the NPZ.")

    names = list(feature_names)
    original_count = len(names)
    name_set = set(names)

    mask = np.ones(original_count, dtype=bool)
    missing_allow: List[str] = []
    missing_deny: List[str] = []

    if allow:
        allow_set = set(allow)
        mask = np.array([n in allow_set for n in names], dtype=bool)
        missing_allow = sorted(allow_set - name_set)

    if deny:
        deny_set = set(deny)
        missing_deny = sorted(deny_set - name_set)
        mask &= np.array([n not in deny_set for n in names], dtype=bool)

    kept_names = [n for n, keep in zip(names, mask) if keep]
    dropped_names = [n for n, keep in zip(names, mask) if not keep]

    if not kept_names:
        raise SystemExit("Feature filtering removed all columns; adjust --allow/--deny.")

    X_filtered = X[:, mask]

    meta = {
        "allow": allow,
        "deny": deny,
        "original_feature_count": original_count,
        "kept_feature_count": len(kept_names),
        "dropped_feature_count": len(dropped_names),
        "dropped_features": dropped_names,
        "applied_by": "eval_kfold.py",
    }
    if missing_allow:
        meta["missing_from_allow"] = missing_allow
    if missing_deny:
        meta["missing_from_deny"] = missing_deny

    return X_filtered, kept_names, mask, meta


def reproject_to_feature_order(
    X: np.ndarray,
    current_names: Optional[List[str]],
    target_names: Optional[List[str]],
) -> Tuple[np.ndarray, Optional[List[str]], Dict[str, List[str]]]:
    """Align columns of X with target_names, inserting zeros for missing ones."""
    info = {"added": [], "dropped": []}
    if target_names is None or current_names is None or current_names == target_names:
        return X, current_names, info

    target_index = {name: idx for idx, name in enumerate(target_names)}
    current_index = {name: idx for idx, name in enumerate(current_names)}

    reordered = np.zeros((X.shape[0], len(target_names)), dtype=X.dtype)
    for j, name in enumerate(target_names):
        idx = current_index.get(name)
        if idx is None:
            info["added"].append(name)
        else:
            reordered[:, j] = X[:, idx]

    for name in current_names:
        if name not in target_index:
            info["dropped"].append(name)

    return reordered, target_names, info


# ---------- Threshold selection ----------
def pick_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    objective: str = "f1",
    beta: float = 1.0,
    min_precision: Optional[float] = None,
) -> float:
    """
    Pick a threshold from the PR curve according to an objective.
    Supported: 'f1', 'fbeta' (with beta), or 'fixed_05' (just 0.5).
    If min_precision is set, only consider points where precision >= min_precision.
    """
    if objective == "fixed_05":
        return 0.5

    # Use precision_recall_curve's thresholds grid
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Add endpoints so threshold indexing stays aligned
    thresholds = np.r_[0.0, thresholds, 1.0]
    precisions = np.r_[precisions[0], precisions, precisions[-1]]
    recalls = np.r_[recalls[0], recalls, recalls[-1]]

    if min_precision is not None:
        mask = precisions >= min_precision
        if not mask.any():
            # fallback: best overall even if precision constraint fails
            mask = np.ones_like(precisions, dtype=bool)
    else:
        mask = np.ones_like(precisions, dtype=bool)

    if objective == "f1":
        denom = precisions + recalls
        f1 = np.zeros_like(denom)
        nz = denom > 0
        f1[nz] = 2 * precisions[nz] * recalls[nz] / denom[nz]
        f1[~mask] = -1.0
        best = int(np.nanargmax(f1))
        return float(thresholds[best])

    if objective == "fbeta":
        beta2 = beta * beta
        denom = beta2 * precisions + recalls
        score = np.zeros_like(denom)
        nz = denom > 0
        score[nz] = (1 + beta2) * precisions[nz] * recalls[nz] / denom[nz]
        score[~mask] = -1.0
        best = int(np.nanargmax(score))
        return float(thresholds[best])

    # default fallback
    return 0.5


# ---------- CV Runner ----------
@dataclass
class CVResult:
    C: float
    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    mean_threshold: float


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    Cs: List[float],
    n_splits: int = 5,
    seed: int = 42,
    objective: str = "f1",
    beta: float = 1.0,
    min_precision: Optional[float] = None,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[CVResult, Dict[float, CVResult]]:
    """
    For each C in Cs, run StratifiedKFold CV, pick best threshold per fold,
    aggregate metrics. Return the best-C result and a map of C→result.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: Dict[float, CVResult] = {}

    for C in Cs:
        fold_rows = []
        thrs = []
        for tr, va in skf.split(X, y):
            Xtr, Xva = X[tr], X[va]
            ytr, yva = y[tr], y[va]

            clf = make_clf(args, C) if args is not None else LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
            clf.fit(Xtr, ytr)
            yprob = clf.predict_proba(Xva)[:, 1]

            thr = pick_threshold(yva, yprob, objective=objective, beta=beta, min_precision=min_precision)
            thrs.append(thr)

            yhat = (yprob >= thr).astype(int)

            fold_rows.append(dict(
                f1=f1_score(yva, yhat),
                pr_auc=average_precision_score(yva, yprob),
                roc_auc=roc_auc_score(yva, yprob),
                brier=brier_score_loss(yva, yprob),
                precision=precision_score(yva, yhat, zero_division=0),
                recall=recall_score(yva, yhat, zero_division=0),
                thr=thr,
            ))

        # aggregate
        def agg(key):
            vals = np.array([r[key] for r in fold_rows], float)
            return float(vals.mean())

        mean_metrics = {k: agg(k) for k in ["f1","pr_auc","roc_auc","brier","precision","recall","thr"]}
        results[C] = CVResult(
            C=C,
            fold_metrics=fold_rows,
            mean_metrics=mean_metrics,
            mean_threshold=mean_metrics["thr"],
        )

    # pick best C by primary metric (objective)
    def score_for(cvres: CVResult) -> float:
        if objective in ("f1", "fbeta"):
            return cvres.mean_metrics["f1"]
        # fallback: PR-AUC
        return cvres.mean_metrics["pr_auc"]

    bestC = max(results.values(), key=score_for)
    return bestC, results


# ---------- Train final + optional test ----------
def train_final_and_eval(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]],
    bestC: float,
    final_threshold: float,
    test_npz: Optional[str] = None,
    test_csv: Optional[str] = None,
    save_model: Optional[str] = None,
    save_report: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    feature_fingerprint: Optional[str] = None,
    featurization_meta: Optional[Dict] = None,
    tag: Optional[str] = None,
    feature_mask: Optional[np.ndarray] = None,
    feature_config_dict: Optional[Dict[str, Any]] = None,
    feature_filter_meta: Optional[Dict[str, Any]] = None,
    predictions_csv: Optional[str] = None,
    extra_report_fields: Optional[Dict[str, Any]] = None,
):
    clf = make_clf(args, bestC) if args is not None else LogisticRegression(max_iter=1000, class_weight="balanced", C=bestC)
    clf.fit(X, y)

    report = dict(bestC=bestC, final_threshold=final_threshold)
    report.update({
        "n_train": int(len(y)),
        "feature_names_count": int(len(feature_names)) if feature_names is not None else None,
        "feature_fingerprint": feature_fingerprint,
        "solver": getattr(args, "solver", None),
        "penalty": getattr(args, "penalty", None),
        "standardize": bool(getattr(args, "standardize", False)),
        "class_weight": getattr(args, "class_weight", None),
        "max_iter": getattr(args, "max_iter", None),
        "tag": tag,
        "use_embed_alignment": bool(getattr(args, "use_embed_alignment", False)),
    })
    if feature_config_dict:
        report["feature_filter"] = feature_config_dict
    if feature_filter_meta:
        report["feature_filter_meta"] = feature_filter_meta
    if featurization_meta is not None:
        report["featurization_meta"] = featurization_meta
    if extra_report_fields:
        report.update(extra_report_fields)

    if test_npz:
        t = np.load(test_npz, allow_pickle=True)
        Xt, yt = t["X"], t["y"]
        test_feature_names = list(t["feature_names"]) if "feature_names" in t else None

        Xt, _, align_info = reproject_to_feature_order(Xt, test_feature_names, feature_names)
        if align_info["added"]:
            print(f"[info] Test NPZ missing columns filled with zeros: {align_info['added']}")
        if align_info["dropped"]:
            print(f"[info] Test NPZ had extra columns dropped: {align_info['dropped']}")

        yprob = clf.predict_proba(Xt)[:, 1]
        yhat = (yprob >= final_threshold).astype(int)

        row_metadata: List[Dict[str, Any]] = []
        answer_types: List[str] = []
        report["test"] = dict(
            f1=float(f1_score(yt, yhat)),
            pr_auc=float(average_precision_score(yt, yprob)),
            roc_auc=float(roc_auc_score(yt, yprob)),
            brier=float(brier_score_loss(yt, yprob)),
            precision=float(precision_score(yt, yhat, zero_division=0)),
            recall=float(recall_score(yt, yhat, zero_division=0)),
        )
        if test_csv:
            try:
                with open(test_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx_row, row in enumerate(reader):
                        val = row.get("answer_type")
                        answer_types.append(val.strip() if val else "unknown")
                        meta_entry = {
                            "row_id": idx_row,
                            "q": None,
                            "a": None,
                            "answer_type": answer_types[-1],
                        }
                        q_raw = row.get("q")
                        a_raw = row.get("a")
                        try:
                            if q_raw:
                                meta_entry["q"] = json.loads(q_raw)
                        except Exception:
                            meta_entry["q"] = q_raw
                        try:
                            if a_raw:
                                meta_entry["a"] = json.loads(a_raw)
                        except Exception:
                            meta_entry["a"] = a_raw
                        row_metadata.append(meta_entry)
                if len(answer_types) == len(yt):
                    per_cat = {}
                    for cat in sorted(set(answer_types)):
                        idxs = [i for i, c in enumerate(answer_types) if c == cat]
                        yi = yt[idxs]
                        ypi = yprob[idxs]
                        yhi = yhat[idxs]
                        per_cat[cat] = dict(
                            support=len(idxs),
                            precision=float(precision_score(yi, yhi, zero_division=0)),
                            recall=float(recall_score(yi, yhi, zero_division=0)),
                            f1=float(f1_score(yi, yhi, zero_division=0)),
                            pr_auc=float(average_precision_score(yi, ypi)) if len(set(yi)) > 1 else None,
                        )
                    report["test"]["by_answer_type"] = per_cat
                else:
                    report["test"]["by_answer_type_warning"] = (
                        f"Row count mismatch: csv={len(answer_types)} npz={len(yt)}; skipping breakdown"
                    )
            except Exception as exc:
                report["test"]["by_answer_type_warning"] = f"Failed to compute breakdown: {exc}"

        # Optionally dump detailed predictions
        if predictions_csv:
            if not test_csv:
                raise SystemExit("--save-predictions requires --test-csv so we can recover question/answer metadata")

            lr_inner = _extract_lr(clf)
            contributions: Optional[np.ndarray] = None
            coef: Optional[np.ndarray] = None

            if lr_inner is not None and hasattr(lr_inner, "coef_"):
                coef = lr_inner.coef_.ravel()
                Xt_for_lr = Xt
                try:
                    if hasattr(clf, "__getitem__"):
                        transformer = clf[:-1]
                        if hasattr(transformer, "transform"):
                            Xt_for_lr = transformer.transform(Xt)
                except Exception:
                    Xt_for_lr = Xt
                try:
                    contributions = Xt_for_lr * coef
                except Exception:
                    contributions = None

            fields = [
                "row_id",
                "question",
                "answer",
                "answer_type",
                "y_true",
                "y_pred",
                "probability",
                "top_features",
            ]
            Path(predictions_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(predictions_csv, "w", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for idx_sample in range(len(yt)):
                    meta = row_metadata[idx_sample] if idx_sample < len(row_metadata) else {"row_id": idx_sample}
                    top_feats: List[Dict[str, Any]] = []
                    if contributions is not None and feature_names is not None and coef is not None:
                        contrib_row = contributions[idx_sample]
                        feature_values = Xt[idx_sample]
                        order = np.argsort(np.abs(contrib_row))[-10:][::-1]
                        for j in order:
                            top_feats.append({
                                "feature": feature_names[j],
                                "value": float(feature_values[j]),
                                "coefficient": float(coef[j]),
                                "contribution": float(contrib_row[j]),
                            })
                    writer.writerow({
                        "row_id": meta.get("row_id", idx_sample),
                        "question": meta.get("q", ""),
                        "answer": meta.get("a", ""),
                        "answer_type": meta.get("answer_type", answer_types[idx_sample] if idx_sample < len(answer_types) else ""),
                        "y_true": int(yt[idx_sample]),
                        "y_pred": int(yhat[idx_sample]),
                        "probability": float(yprob[idx_sample]),
                        "top_features": json.dumps(top_feats),
                    })

    if save_model:
        os.makedirs(os.path.dirname(save_model) or ".", exist_ok=True)
        payload = dict(
            sklearn_lr=clf,
            feature_names=feature_names,
            threshold=final_threshold,
            C=bestC,
            feature_fingerprint=feature_fingerprint,
            featurization_meta=featurization_meta,  # e.g., feature_config / metrics_config / boolean_allowlist
            solver=getattr(args, "solver", None),
            penalty=getattr(args, "penalty", None),
            standardize=bool(getattr(args, "standardize", False)),
            class_weight=getattr(args, "class_weight", None),
            max_iter=getattr(args, "max_iter", None),
            tag=tag,
            use_embed_alignment=bool(getattr(args, "use_embed_alignment", False)),
        )
        if feature_config_dict:
            payload["feature_config"] = feature_config_dict
        with open(save_model, "wb") as f:
            pickle.dump(payload, f)

    if save_report:
        os.makedirs(os.path.dirname(save_report) or ".", exist_ok=True)
        with open(save_report, "w") as f:
            json.dump(report, f, indent=2)

    return clf, report


# ---------- CLI ----------
def parse_Cs(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def make_clf(args: argparse.Namespace, C: float):
    cw = None if args.class_weight == "none" else "balanced"
    solver = args.solver
    penalty = args.penalty  # "l2", "l1", or "none"

    # Normalize incompatible combos:
    # liblinear: supports l1 or l2 only
    if solver == "liblinear" and penalty == "none":
        penalty = "l2"
    # lbfgs: supports l2 or none (not l1)
    if solver == "lbfgs" and penalty == "l1":
        penalty = "l2"
    # saga: supports l1/l2/none (OK)

    lr = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=cw,
        C=C,
        solver=solver,
        penalty=penalty,
    )
    if args.standardize:
        return make_pipeline(StandardScaler(), lr)
    return lr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True, help="Processed features: np.savez(..., X, y, feature_names)")
    ap.add_argument("--test-npz", default=None, help="Optional held-out test NPZ for final check")
    ap.add_argument("--test-csv", default=None, help="Optional CSV aligned with --test-npz (used for metadata breakdown)")
    ap.add_argument("--Cs", default="0.1,0.3,1,3,10", help="Comma-separated Cs to try")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", default="f1", choices=["f1","fbeta","fixed_05"])
    ap.add_argument("--beta", type=float, default=1.0, help="Only for fbeta")
    ap.add_argument("--min-precision", type=float, default=None, help="Require precision >= this during threshold pick")
    ap.add_argument("--save-model", default=None, help="Pickle path to save fitted LR + threshold")
    ap.add_argument("--save-report", default=None, help="JSON path to save CV summary + test metrics")
    ap.add_argument("--save-predictions", default=None, help="CSV path to dump per-row test predictions (requires --test-csv)")
    ap.add_argument("--solver", default="lbfgs", choices=["lbfgs","liblinear","saga"])
    ap.add_argument("--penalty", default="l2", choices=["l2","l1","none"])
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--standardize", action="store_true", help="Apply StandardScaler before LR")
    ap.add_argument("--class-weight", default="balanced", choices=["balanced","none"])
    ap.add_argument("--use-embed-alignment", action="store_true",
                    help="Flag only recorded in artifacts: indicates features were computed with MiniLM embedding alignment enabled upstream.")
    ap.add_argument("--featurization-meta", default=None,
                    help="Optional JSON with featurization settings used upstream (feature_config, metrics_config, boolean_allowlist). Will be embedded in the saved model payload.")
    ap.add_argument("--tag", default=None,
                    help="Optional freeform tag to store in the saved payload/report (e.g., a data/version label).")
    ap.add_argument("--allow", nargs="*", default=None,
                    help="Exact feature names to keep (applied after loading --train-npz). If provided, only these columns remain.")
    ap.add_argument("--deny", nargs="*", default=None,
                    help=("Exact feature names to drop (applied after loading --train-npz). "
                          "Columns listed here are removed before training/eval."))
    args = ap.parse_args()

    npz = np.load(args.train_npz, allow_pickle=True)
    X, y = npz["X"], npz["y"]
    # Feature names & fingerprint
    feature_names = list(npz["feature_names"]) if "feature_names" in npz else None

    def _normalize_feature_list(values: Optional[List[str]]) -> List[str]:
        if not values:
            return []
        cleaned = [v.strip() for v in values if v and v.strip()]
        return list(dict.fromkeys(cleaned))

    allow_list = _normalize_feature_list(args.allow)
    deny_list = _normalize_feature_list(args.deny)
    feature_mask: Optional[np.ndarray] = None
    filter_meta: Optional[Dict[str, Any]] = None
    feature_config_dict: Optional[Dict[str, Any]] = None

    if allow_list or deny_list:
        if feature_names is None:
            raise SystemExit("--allow/--deny requires feature_names in the NPZ")
        X, feature_names, feature_mask, filter_meta = apply_feature_filters(
            X,
            feature_names,
            allow=allow_list,
            deny=deny_list,
        )
        feature_config_dict = {
            "allowlist": allow_list,
            "denylist": deny_list,
            "allow_patterns": [],
            "deny_patterns": [],
        }
        if filter_meta:
            kept = filter_meta.get("kept_feature_count", X.shape[1])
            orig = filter_meta.get("original_feature_count", X.shape[1])
            dropped = filter_meta.get("dropped_feature_count", 0)
            print(
                f"Applied feature filter: kept {kept}/{orig} columns (dropped {dropped})."
            )
            print(f"  allowlist: {allow_list or '[]'}")
            print(f"  denylist: {deny_list or '[]'}")
            print(f"  resulting feature set ({len(feature_names)} columns):")
            preview_limit = 20
            if len(feature_names) > preview_limit:
                preview = feature_names[:preview_limit]
                print(f"    {preview} ... (and {len(feature_names) - preview_limit} more)")
            else:
                print(f"    {feature_names}")
            if filter_meta.get("missing_from_allow"):
                print(f"  [warn] Missing from allow: {filter_meta['missing_from_allow']}")
        feature_mask = None
    else:
        feature_mask = None

    def _fingerprint(names):
        if not names: return None
        s = "|".join(str(x) for x in names).encode("utf-8")
        return hashlib.sha256(s).hexdigest()
    feature_fp = _fingerprint(feature_names)

    # Optional featurization metadata (keeps train/infer consistent)
    featurization_meta = None
    if args.featurization_meta:
        try:
            with open(args.featurization_meta, "r") as f:
                featurization_meta = json.load(f)
        except Exception as e:
            print(f"[warn] Failed to load --featurization-meta: {e}")
            featurization_meta = None

    if filter_meta:
        base_meta = dict(featurization_meta) if isinstance(featurization_meta, dict) else {}
        base_meta["post_filter_feature_config"] = filter_meta
        featurization_meta = base_meta

    bestCres, allres = run_cv(
        X, y,
        Cs=parse_Cs(args.Cs),
        n_splits=args.n_splits,
        seed=args.seed,
        objective=args.objective,
        beta=args.beta,
        min_precision=args.min_precision,
        args=args,
    )

    print("CV results by C:")
    for C, res in sorted(allres.items(), key=lambda kv: kv[0]):
        m = res.mean_metrics
        print(f"  C={C:>5}: F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}  ROC-AUC={m['roc_auc']:.3f}  "
              f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  Thr~{m['thr']:.3f}")

    cv_summary = {
        "n_splits": args.n_splits,
        "objective": args.objective,
        "beta": args.beta if args.objective == "fbeta" else None,
        "min_precision": args.min_precision,
        "seed": args.seed,
        "grid": {
            str(C): {
                "mean_metrics": res.mean_metrics,
                "mean_threshold": res.mean_threshold,
                "fold_metrics": res.fold_metrics,
            }
            for C, res in sorted(allres.items(), key=lambda kv: kv[0])
        },
        "selected": {
            "C": bestCres.C,
            "mean_threshold": bestCres.mean_threshold,
            "mean_metrics": bestCres.mean_metrics,
            "fold_metrics": bestCres.fold_metrics,
        },
    }

    print(f"\nSelected C={bestCres.C} with mean threshold ~{bestCres.mean_threshold:.3f}")
    clf, report = train_final_and_eval(
        X, y, feature_names,
        bestCres.C, bestCres.mean_threshold,
        test_npz=args.test_npz,
        test_csv=args.test_csv,
        save_model=args.save_model,
        save_report=args.save_report,
        args=args,
        feature_fingerprint=feature_fp,
        featurization_meta=featurization_meta,
        tag=args.tag,
        feature_mask=feature_mask,
        feature_config_dict=feature_config_dict,
        feature_filter_meta=filter_meta,
        predictions_csv=args.save_predictions,
        extra_report_fields={"cv": cv_summary},
    )

    if "test" in report:
        t = report["test"]
        print("\nTEST:")
        print(f"  F1={t['f1']:.3f}  PR-AUC={t['pr_auc']:.3f}  ROC-AUC={t['roc_auc']:.3f}  "
              f"Prec={t['precision']:.3f}  Rec={t['recall']:.3f}")
        by_cat = t.get("by_answer_type")
        if isinstance(by_cat, dict):
            print("  Breakdown by answer_type:")
            for cat, metrics in sorted(by_cat.items()):
                support = metrics.get("support", 0)
                prec = metrics.get("precision")
                rec = metrics.get("recall")
                f1 = metrics.get("f1")
                def _fmt(v):
                    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
                print(
                    f"    {cat or 'unknown'}: support={support}  precision={_fmt(prec)}  "
                    f"recall={_fmt(rec)}  f1={_fmt(f1)}"
                )
        elif "by_answer_type_warning" in t:
            print(f"  [warn] {t['by_answer_type_warning']}")

    # Show top coefficients if feature names available
    if feature_names is not None:
        lr_inner = _extract_lr(clf)
        if lr_inner is None or not hasattr(lr_inner, "coef_"):
            print("\n(Skipping coefficient print — no bare LogisticRegression found inside the estimator.)")
        else:
            coefs = lr_inner.coef_.ravel()
            order = np.argsort(coefs)
            print("\nTop + features:")
            for i in order[-10:][::-1]:
                print(f"  {feature_names[i]}: {coefs[i]:+.4f}")
            print("\nTop − features:")
            for i in order[:10]:
                print(f"  {feature_names[i]}: {coefs[i]:+.4f}")


if __name__ == "__main__":
    main()
