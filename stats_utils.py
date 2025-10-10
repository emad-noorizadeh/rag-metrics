"""Shared helpers for dataset statistics (label counts, answer types)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    # Reuse the canonical allowlist when available.
    from scripts.analyze_predictions import ALLOWED_ANSWER_TYPES as _ALLOWED_ANSWER_TYPES
except Exception:  # pragma: no cover - fallback when dependency missing
    _ALLOWED_ANSWER_TYPES = {"list", "fact", "abstain", "numeric", "inference", "comparison"}

ACCEPTABLE_ANSWER_TYPES = {t.lower() for t in _ALLOWED_ANSWER_TYPES}


def compute_dataset_stats(meta: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(meta)
    positives = sum(1 for row in meta if int(row.get("y", 0)) == 1)
    negatives = total - positives
    stats: Dict[str, Any] = {
        "total_examples": total,
        "positives": positives,
        "negatives": negatives,
    }

    if any((row.get("answer_type") or "").strip() for row in meta):
        counts: Dict[str, int] = {}
        for row in meta:
            if int(row.get("y", 0)) != 1:
                continue
            raw = (row.get("answer_type") or "").strip()
            if not raw:
                bucket = "unknown"
            else:
                norm = raw.lower()
                bucket = norm if norm in ACCEPTABLE_ANSWER_TYPES else "other"
            counts[bucket] = counts.get(bucket, 0) + 1
        if counts:
            stats["answer_type_positive_counts"] = dict(sorted(counts.items()))
            stats["allowed_answer_types"] = sorted(ACCEPTABLE_ANSWER_TYPES)

    return stats


def save_dataset_stats(meta: List[Dict[str, Any]], out_prefix: str) -> Optional[str]:
    stats = compute_dataset_stats(meta)
    if not stats:
        return None

    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, f"stats_{os.path.basename(out_prefix)}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    return stats_path

