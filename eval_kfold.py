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
import argparse, json, math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

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
    save_model: Optional[str] = None,
    save_report: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    feature_fingerprint: Optional[str] = None,
    featurization_meta: Optional[Dict] = None,
    tag: Optional[str] = None,
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
    if featurization_meta is not None:
        report["featurization_meta"] = featurization_meta

    if test_npz:
        t = np.load(test_npz, allow_pickle=True)
        Xt, yt = t["X"], t["y"]
        yprob = clf.predict_proba(Xt)[:, 1]
        yhat = (yprob >= final_threshold).astype(int)
        report["test"] = dict(
            f1=float(f1_score(yt, yhat)),
            pr_auc=float(average_precision_score(yt, yprob)),
            roc_auc=float(roc_auc_score(yt, yprob)),
            brier=float(brier_score_loss(yt, yprob)),
            precision=float(precision_score(yt, yhat, zero_division=0)),
            recall=float(recall_score(yt, yhat, zero_division=0)),
        )

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
    ap.add_argument("--Cs", default="0.1,0.3,1,3,10", help="Comma-separated Cs to try")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--objective", default="f1", choices=["f1","fbeta","fixed_05"])
    ap.add_argument("--beta", type=float, default=1.0, help="Only for fbeta")
    ap.add_argument("--min-precision", type=float, default=None, help="Require precision >= this during threshold pick")
    ap.add_argument("--save-model", default=None, help="Pickle path to save fitted LR + threshold")
    ap.add_argument("--save-report", default=None, help="JSON path to save CV summary + test metrics")
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
    args = ap.parse_args()

    npz = np.load(args.train_npz, allow_pickle=True)
    X, y = npz["X"], npz["y"]
    # Feature names & fingerprint
    feature_names = list(npz["feature_names"]) if "feature_names" in npz else None
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

    print(f"\nSelected C={bestCres.C} with mean threshold ~{bestCres.mean_threshold:.3f}")
    clf, report = train_final_and_eval(
        X, y, feature_names,
        bestCres.C, bestCres.mean_threshold,
        test_npz=args.test_npz,
        save_model=args.save_model,
        save_report=args.save_report,
        args=args,
        feature_fingerprint=feature_fp,
        featurization_meta=featurization_meta,
        tag=args.tag,
    )

    if "test" in report:
        t = report["test"]
        print("\nTEST:")
        print(f"  F1={t['f1']:.3f}  PR-AUC={t['pr_auc']:.3f}  ROC-AUC={t['roc_auc']:.3f}  "
              f"Prec={t['precision']:.3f}  Rec={t['recall']:.3f}")

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