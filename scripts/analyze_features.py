#!/usr/bin/env python3
"""Offline feature diagnostics.

Usage:
  python scripts/analyze_features.py --npz data/processed/train_v3.npz \
      --model artifacts/lr_model_v1.pkl --out artifacts/feature_report.json

The script loads the feature matrix, computes basic statistics, inspects the
logistic-regression coefficients saved by ``eval_kfold.py``, and highlights
strongly correlated features / zero-variance columns to help with pruning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def _extract_lr(est) -> Optional[LogisticRegression]:
    if isinstance(est, LogisticRegression):
        return est
    named = getattr(est, "named_steps", None)
    if named:
        for _, step in reversed(list(named.items())):
            if isinstance(step, LogisticRegression):
                return step
    steps = getattr(est, "steps", None)
    if steps:
        for _, step in reversed(steps):
            if isinstance(step, LogisticRegression):
                return step
    return None


def load_model(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "sklearn_lr" in payload:
        model = payload["sklearn_lr"]
        threshold = payload.get("threshold")
        meta = {k: payload.get(k) for k in payload if k not in {"sklearn_lr"}}
        return model, threshold, meta
    # assume bare estimator
    return payload, None, {}


def zero_variance_features(X: np.ndarray, feature_names: List[str]) -> List[str]:
    variances = X.var(axis=0)
    return [feature_names[i] for i, v in enumerate(variances) if math.isclose(float(v), 0.0, abs_tol=1e-12)]


def feature_correlations(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    # Pearson correlation of each feature with the label.
    corrs: List[Tuple[str, float]] = []
    for i in range(X.shape[1]):
        x = X[:, i]
        if np.allclose(x.std(), 0):
            corrs.append((feature_names[i], 0.0))
            continue
        c = np.corrcoef(x, y)[0, 1]
        if not np.isfinite(c):
            c = 0.0
        corrs.append((feature_names[i], float(c)))
    return corrs


def high_correlation_pairs(X: np.ndarray, feature_names: List[str], threshold: float = 0.9) -> List[Tuple[str, str, float]]:
    if X.shape[1] > 200:
        return []  # avoid giant matrices
    corr = np.corrcoef(X, rowvar=False)
    pairs = []
    n = corr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            val = corr[i, j]
            if np.isfinite(val) and abs(val) >= threshold:
                pairs.append((feature_names[i], feature_names[j], float(val)))
    return sorted(pairs, key=lambda x: -abs(x[2]))


def summarize_coefficients(lr: LogisticRegression, feature_names: List[str], top_k: int = 15):
    coef = lr.coef_.ravel()
    order = np.argsort(coef)
    top_pos = [(feature_names[i], float(coef[i])) for i in order[-top_k:][::-1]]
    top_neg = [(feature_names[i], float(coef[i])) for i in order[:top_k]]
    return top_pos, top_neg


def main():
    ap = argparse.ArgumentParser(description="Feature diagnostics for trained models")
    ap.add_argument("--npz", required=True, help="npz with X, y, feature_names (output of data_processing)")
    ap.add_argument("--model", help="Pickle saved by eval_kfold.py (contains sklearn_lr)")
    ap.add_argument("--out", help="Optional JSON report path")
    ap.add_argument("--corr-threshold", type=float, default=0.9, help="Absolute correlation threshold for pair reporting")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X = data["X"]
    y = data.get("y")
    feature_names = list(data["feature_names"]) if "feature_names" in data else [f"f{i}" for i in range(X.shape[1])]

    model = threshold = model_meta = None
    lr = None
    if args.model:
        model, threshold, model_meta = load_model(args.model)
        lr = _extract_lr(model)

    zvar = zero_variance_features(X, feature_names)
    corr_with_y = feature_correlations(X, y, feature_names) if y is not None else []
    corr_with_y_sorted = sorted(corr_with_y, key=lambda kv: -abs(kv[1]))
    hi_pairs = high_correlation_pairs(X, feature_names, threshold=args.corr_threshold)

    top_pos = top_neg = None
    if lr is not None and hasattr(lr, "coef_"):
        top_pos, top_neg = summarize_coefficients(lr, feature_names)

    report = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "zero_variance_features": zvar,
        "top_corr_with_y": corr_with_y_sorted[:20],
        "high_correlation_pairs": hi_pairs[:20],
        "model_threshold": threshold,
        "model_meta": model_meta,
        "top_positive_coefficients": top_pos,
        "top_negative_coefficients": top_neg,
    }

    print("Feature statistics:")
    print(json.dumps({k: v for k, v in report.items() if k not in {"model_meta"}}, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved feature report to {args.out}")


if __name__ == "__main__":
    main()
