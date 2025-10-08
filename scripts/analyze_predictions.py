#!/usr/bin/env python3
"""Analyze prediction CSV exported by eval_kfold.

Outputs aggregated diagnostics:
- Outcome counts (TP/FP/FN/TN) and answer_type breakdown.
- Feature-level contribution stats per outcome (counts, averages, sign split).
- Prefix/group summaries (e.g., entity_match.*, qr_alignment.*).
- TP vs FN impact matrix with normalised scores and ranking files for quick triage.

Example:
    python scripts/analyze_predictions.py \
        --predictions artifacts/predictions_v1.csv \
        --out-json artifacts/predictions_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

ALLOWED_ANSWER_TYPES = {"list", "fact", "abstain", "numeric", "inference", "comparison"}


@dataclass
class FeatureContribution:
    positive: int = 0
    negative: int = 0
    positive_total_contribution: float = 0.0
    negative_total_contribution: float = 0.0

    def add(self, contribution: float) -> None:
        if contribution >= 0:
            self.positive += 1
            self.positive_total_contribution += contribution
        else:
            self.negative += 1
            self.negative_total_contribution += contribution

    def to_dict(self) -> Dict[str, float]:
        total = self.positive + self.negative
        net = self.positive_total_contribution + self.negative_total_contribution
        return {
            "total_count": total,
            "positive_count": self.positive,
            "negative_count": self.negative,
            "positive_contribution_sum": self.positive_total_contribution,
            "negative_contribution_sum": self.negative_total_contribution,
            "net_contribution_sum": net,
        }


@dataclass
class OutcomeContribution:
    total_contribution: float = 0.0
    count: int = 0
    positive: int = 0
    negative: int = 0
    answer_type_counts: Dict[str, int] = field(default_factory=dict)

    def add(self, contribution: float, answer_type: str) -> None:
        self.total_contribution += contribution
        self.count += 1
        if contribution >= 0:
            self.positive += 1
        else:
            self.negative += 1
        self.answer_type_counts[answer_type] = self.answer_type_counts.get(answer_type, 0) + 1

    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_contribution / self.count

    def to_dict(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "average_contribution": round(self.average(), 6),
            "positive_count": self.positive,
            "negative_count": self.negative,
            "by_answer_type": self.answer_type_counts,
        }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze eval_kfold predictions")
    ap.add_argument("--predictions", required=True, help="CSV produced by eval_kfold --save-predictions")
    ap.add_argument("--out-json", default=None, help="Optional JSON report path")
    ap.add_argument("--top-n", type=int, default=10,
                    help="How many feature contributions per row (default aligns with exporter)")
    return ap.parse_args()


def load_predictions(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
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


def analyze(rows: List[Dict[str, object]], top_n: int) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "total": len(rows),
        "by_outcome": {},
        "by_answer_type": {},
        "feature_impact": {"FP": {}, "FN": {}, "TP": {}},
    }

    tp_scores: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    fn_scores: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0])
    feature_outcome_stats: Dict[str, Dict[str, OutcomeContribution]] = defaultdict(lambda: defaultdict(OutcomeContribution))
    global_feature_signs: Dict[str, FeatureContribution] = defaultdict(FeatureContribution)
    prefix_stats: Dict[str, Dict[str, OutcomeContribution]] = defaultdict(lambda: defaultdict(OutcomeContribution))

    for row in rows:
        outcome = classify_outcome(row["y_true"], row["y_pred"])
        raw_answer_type = row.get("answer_type", "unknown") or "unknown"
        answer_type = raw_answer_type.lower()
        if answer_type not in ALLOWED_ANSWER_TYPES:
            answer_type = "other"

        summary["by_outcome"].setdefault(outcome, 0)
        summary["by_outcome"][outcome] += 1

        by_type = summary["by_answer_type"].setdefault(answer_type, {"total": 0, "outcomes": {}})
        by_type["total"] += 1
        by_type["outcomes"].setdefault(outcome, 0)
        by_type["outcomes"][outcome] += 1

        if outcome not in {"FP", "FN", "TP"}:
            continue

        feats = row.get("top_features", [])[:top_n]
        for feat in feats:
            name = feat.get("feature")
            if not name:
                continue
            contribution = float(feat.get("contribution", 0.0))

            # Per-outcome feature stats for quick filtering
            impact_store = summary["feature_impact"].setdefault(outcome, {})
            impact_entry = impact_store.setdefault(name, FeatureContribution())
            impact_entry.add(contribution)

            # Detailed outcome stats per feature
            feature_outcome_stats[name][outcome].add(contribution, answer_type)

            # Global sign summary (coefficient/sign style)
            global_feature_signs[name].add(contribution)

            # Prefix / feature-family summary
            prefix = name.split(".", 1)[0]
            prefix_stats[prefix][outcome].add(contribution, answer_type)

            if outcome == "TP":
                agg = tp_scores[name]
                agg[0] += contribution
                agg[1] += 1
            elif outcome == "FN":
                agg = fn_scores[name]
                agg[0] += contribution
                agg[1] += 1

    # Finalise dictionaries
    for outcome, feats in summary["feature_impact"].items():
        summary["feature_impact"][outcome] = {
            feature: contrib.to_dict() for feature, contrib in feats.items()
        }

    summary["feature_outcome_stats"] = {
        feature: {outcome: stats.to_dict() for outcome, stats in outcomes.items()}
        for feature, outcomes in feature_outcome_stats.items()
    }

    summary["feature_sign_summary"] = {
        feature: contrib.to_dict() for feature, contrib in global_feature_signs.items()
    }

    summary["feature_group_summary"] = {
        prefix: {outcome: stats.to_dict() for outcome, stats in outcomes.items()}
        for prefix, outcomes in prefix_stats.items()
    }

    impact_matrix, ranking = compute_impact_matrix(tp_scores, fn_scores)
    summary["impact_matrix"] = impact_matrix
    summary["impact_ranking"] = ranking

    return summary


def compute_impact_matrix(
    tp_scores: Dict[str, List[float]],
    fn_scores: Dict[str, List[float]],
) -> Tuple[Dict[str, Dict[str, float]], List[Tuple[str, float]]]:
    if not tp_scores and not fn_scores:
        return {}, []

    tp_avg = {name: (vals[0] / vals[1]) if vals[1] else 0.0 for name, vals in tp_scores.items()}
    fn_avg = {name: (vals[0] / vals[1]) if vals[1] else 0.0 for name, vals in fn_scores.items()}

    tp_max = max((abs(v) for v in tp_avg.values()), default=1.0)
    fn_max = max((abs(v) for v in fn_avg.values()), default=1.0)

    matrix: Dict[str, Dict[str, float]] = {}
    rankings: List[Tuple[str, float]] = []
    all_features = set(tp_avg) | set(fn_avg)
    for name in sorted(all_features):
        avg_tp = tp_avg.get(name, 0.0)
        avg_fn = fn_avg.get(name, 0.0)
        norm_tp = avg_tp / tp_max if tp_max else 0.0
        norm_fn = abs(avg_fn) / fn_max if fn_max else 0.0  # penalise FN magnitude
        score = norm_tp - norm_fn
        matrix[name] = {
            "avg_tp": round(avg_tp, 6),
            "avg_fn": round(avg_fn, 6),
            "norm_tp": round(norm_tp, 4),
            "norm_fn": round(norm_fn, 4),
            "score": round(score, 4),
        }
        rankings.append((name, score))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return matrix, rankings


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    if not predictions:
        raise SystemExit("No prediction rows found.")

    summary = analyze(predictions, top_n=args.top_n)

    print(json.dumps(summary, indent=2))

    out_json = args.out_json
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved analysis to {out_json}")

    impact_matrix = summary.get("impact_matrix") or {}
    impact_ranking = summary.get("impact_ranking") or []
    if impact_matrix:
        matrix_path = (Path(out_json).with_name("impact_matrix_TPFN.csv")
                       if out_json else Path("impact_matrix_TPFN.csv"))
        with open(matrix_path, "w", encoding="utf-8") as f:
            f.write("feature,avg_tp,avg_fn,norm_tp,norm_fn,score\n")
            for feature, scores in impact_matrix.items():
                f.write(
                    f"{feature},{scores['avg_tp']:.6f},{scores['avg_fn']:.6f},{scores['norm_tp']:.4f},{scores['norm_fn']:.4f},{scores['score']:.4f}\n"
                )
        print(f"Saved TP/FN impact matrix → {matrix_path}")

        ranking_path = (Path(out_json).with_name("impact_ranking_TPFN.csv")
                        if out_json else Path("impact_ranking_TPFN.csv"))
        with open(ranking_path, "w", encoding="utf-8") as f:
            f.write("feature,score\n")
            for feature, score in impact_ranking:
                f.write(f"{feature},{score:.4f}\n")
        print(f"Saved TP/FN impact ranking → {ranking_path}")
    ranking_path = (Path(out_json).with_name("impact_ranking_TPFN.csv")
                    if out_json else Path("impact_ranking_TPFN.csv"))
    if impact_ranking:
        with open(ranking_path, "w", encoding="utf-8") as f:
            f.write("feature,score\n")
            for feature, score in impact_ranking:
                f.write(f"{feature},{score:.4f}\n")
        print(f"Saved TP/FN impact ranking → {ranking_path}")


if __name__ == "__main__":
    main()
