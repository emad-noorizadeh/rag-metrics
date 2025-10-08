#!/usr/bin/env python3
"""Analyze prediction CSV exported by eval_kfold.

The script groups rows by ground-truth label, prediction outcome (TP/FP/FN/TN),
answer_type, and individual features appearing in the `top_features` column. It
summarizes how often each feature contributes positively or negatively to false
positives/negatives.

Example:
    python scripts/analyze_predictions.py \
        --predictions artifacts/predictions_v1.csv \
        --out-json artifacts/predictions_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class FeatureContribution:
    positive: int = 0
    negative: int = 0
    positive_total_contribution: float = 0.0
    negative_total_contribution: float = 0.0

    def add(self, contribution: float):
        if contribution >= 0:
            self.positive += 1
            self.positive_total_contribution += contribution
        else:
            self.negative += 1
            self.negative_total_contribution += contribution

    def to_dict(self) -> Dict[str, float]:
        return {
            "positive_count": self.positive,
            "negative_count": self.negative,
            "positive_contribution_sum": self.positive_total_contribution,
            "negative_contribution_sum": self.negative_total_contribution,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze eval_kfold predictions")
    ap.add_argument("--predictions", required=True, help="CSV produced by eval_kfold --save-predictions")
    ap.add_argument("--out-json", default=None, help="Optional JSON report path")
    ap.add_argument("--top-n", type=int, default=10, help="How many feature contributions per row (default aligns with exporter)")
    return ap.parse_args()


def load_predictions(path: str) -> List[Dict[str, any]]:
    rows: List[Dict[str, any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "top_features" not in reader.fieldnames:
            raise SystemExit("predictions CSV is missing 'top_features' column")
        for row in reader:
            try:
                top_feats = json.loads(row.get("top_features", "[]")) or []
            except json.JSONDecodeError:
                top_feats = []
            rows.append({
                "row_id": row.get("row_id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "answer_type": row.get("answer_type") or "unknown",
                "y_true": int(row["y_true"]),
                "y_pred": int(row["y_pred"]),
                "probability": float(row["probability"]),
                "top_features": top_feats,
            })
    return rows


def classify_outcome(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    return "TN"


def analyze(rows: List[Dict[str, any]], top_n: int) -> Dict[str, any]:
    summary: Dict[str, any] = {
        "total": len(rows),
        "by_outcome": {},
        "by_answer_type": {},
        "feature_impact": {
            "FP": {},
            "FN": {},
        },
    }

    for row in rows:
        outcome = classify_outcome(row["y_true"], row["y_pred"])
        answer_type = row.get("answer_type", "unknown") or "unknown"
        summary["by_outcome"].setdefault(outcome, 0)
        summary["by_outcome"][outcome] += 1

        by_type = summary["by_answer_type"].setdefault(answer_type, {"total": 0, "outcomes": {}})
        by_type["total"] += 1
        by_type["outcomes"].setdefault(outcome, 0)
        by_type["outcomes"][outcome] += 1

        if outcome not in {"FP", "FN"}:
            continue

        feats = row.get("top_features", [])[:top_n]
        for feat in feats:
            name = feat.get("feature")
            contribution = float(feat.get("contribution", 0.0))
            if not name:
                continue
            store = summary["feature_impact"].setdefault(outcome, {})
            feat_rec = store.setdefault(name, FeatureContribution())
            feat_rec.add(contribution)

    # Convert FeatureContribution objects to dicts
    for outcome, feats in summary["feature_impact"].items():
        summary["feature_impact"][outcome] = {
            name: contrib.to_dict() for name, contrib in feats.items()
        }

    return summary


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    if not predictions:
        raise SystemExit("No prediction rows found.")

    summary = analyze(predictions, top_n=args.top_n)

    print(json.dumps(summary, indent=2))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved analysis to {args.out_json}")


if __name__ == "__main__":
    main()
