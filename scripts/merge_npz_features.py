#!/usr/bin/env python3
"""Merge multiple NPZ feature datasets on the shared feature intersection.

Example:
    python scripts/merge_npz_features.py \
        --inputs data/processed/train_v1.npz data/processed/train_v4.npz \
        --out-npz data/processed/train_merged.npz \
        --out-csv data/processed/train_merged.csv

The script keeps only the features that exist in *all* inputs, reorders each
matrix to that common schema, concatenates the rows, and writes both NPZ and
CSV outputs.
"""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge NPZ feature sets on shared columns")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Two or more NPZ files produced by data_processing.py")
    ap.add_argument("--out-npz", required=True, help="Path for merged NPZ output")
    ap.add_argument("--out-csv", required=True, help="Path for merged CSV output")
    return ap.parse_args()


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data or "feature_names" not in data:
        raise ValueError(f"{path} is missing required arrays (needs X, y, feature_names)")
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    return X, y, feature_names


def write_npz(out_path: str, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, feature_names=np.array(feature_names, dtype=object))


def write_csv(out_path: str, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    header = feature_names + ["y"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(X.shape[0]):
            row_vals = [f"{X[i, j]:.6f}" for j in range(X.shape[1])]
            row_vals.append(str(int(y[i])))
            f.write(",".join(row_vals) + "\n")


def main() -> None:
    args = parse_args()
    if len(args.inputs) < 2:
        raise SystemExit("Provide at least two NPZ inputs")

    datasets = []
    feature_sets = []

    for path in args.inputs:
        X, y, names = load_npz(path)
        datasets.append((X, y, names))
        feature_sets.append(set(names))

    shared_features = set.intersection(*feature_sets)
    if not shared_features:
        raise SystemExit("No common features across the provided NPZ files")

    # Preserve the column order from the first dataset, filtered to the shared set.
    shared_order = [name for name in datasets[0][2] if name in shared_features]

    merged_X = []
    merged_y = []

    for idx, (X, y, names) in enumerate(datasets):
        name_to_idx = {name: i for i, name in enumerate(names)}
        missing = [name for name in shared_order if name not in name_to_idx]
        if missing:
            raise SystemExit(f"Dataset {args.inputs[idx]} missing expected columns: {missing}")
        cols = [name_to_idx[name] for name in shared_order]
        merged_X.append(X[:, cols])
        merged_y.append(y)

    merged_X = np.vstack(merged_X)
    merged_y = np.concatenate(merged_y)

    write_npz(args.out_npz, merged_X, merged_y, shared_order)
    write_csv(args.out_csv, merged_X, merged_y, shared_order)

    print(f"Merged {len(args.inputs)} files → {args.out_npz} ({merged_X.shape[0]} rows, {merged_X.shape[1]} features)")
    print(f"CSV snapshot written → {args.out_csv}")


if __name__ == "__main__":
    main()
