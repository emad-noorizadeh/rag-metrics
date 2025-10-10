#!/usr/bin/env python3

"""Generate dataset statistics JSON from a featurized CSV."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List


def _ensure_repo_root() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root()

from stats_utils import save_dataset_stats


def _parse_label(raw: Any, row_idx: int) -> int:
    if raw is None:
        raise ValueError(f"Row {row_idx}: missing 'y' column")
    text = str(raw).strip()
    if not text:
        raise ValueError(f"Row {row_idx}: empty 'y' value")
    try:
        return int(float(text))
    except ValueError as exc:
        raise ValueError(f"Row {row_idx}: invalid label {raw!r}") from exc


def _load_meta(csv_path: str) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "y" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'y' column produced by data_processing.py")
        for idx, row in enumerate(reader):
            label = _parse_label(row.get("y"), idx + 2)  # +2 to account for header + 1-based rows
            meta.append({
                "y": label,
                "answer_type": row.get("answer_type"),
            })
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate stats_{prefix}.json from a featurized CSV.")
    ap.add_argument("csv", help="Path to the CSV produced by data_processing.py")
    ap.add_argument("--out-prefix", default=None,
                    help="Prefix for the stats file (defaults to CSV path without extension)")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")

    out_prefix = args.out_prefix or os.path.splitext(csv_path)[0]
    meta = _load_meta(csv_path)
    if not meta:
        raise SystemExit("No rows found in CSV; stats not generated.")

    stats_path = save_dataset_stats(meta, out_prefix)
    if not stats_path:
        print("No stats generated (missing labels).")
    else:
        print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
