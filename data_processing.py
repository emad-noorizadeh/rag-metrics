# data_processing.py
from __future__ import annotations
import os, glob, json, argparse
from typing import Any, Dict, List, Iterable, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Use your metrics report
from metric_utils import context_utilization_report_with_entities

# ---------- small helpers ----------

def _is_number(x: Any) -> bool:
    # exclude bools (bool is a subclass of int)
    return (isinstance(x, (int, float))
            and not isinstance(x, bool)
            and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x))))

def _flatten_report(
    rep: Dict[str, Any],
    prefix: str = "",
    include_list_lengths: bool = True,
    include_numeric_list_stats: bool = True,
    include_boolean_allowlist: Iterable[str] = ("inference_likely",),
) -> Dict[str, float]:
    """
    Recursively flatten the report dict, keeping only numeric scalars.
    For lists: optionally add '<key>__len' and (if numeric list) mean/min/max.
    For nested dicts: join path with '.' (e.g., 'entity_match.overall').
    Booleans are excluded by default; allowlist specific ones.
    """
    out: Dict[str, float] = {}

    # allow listed booleans at root-level
    for bkey in include_boolean_allowlist:
        if bkey in rep and isinstance(rep[bkey], bool):
            out[f"{bkey}_bool"] = 1.0 if rep[bkey] else 0.0

    for k, v in rep.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"

        if _is_number(v):
            out[key] = float(v)

        elif isinstance(v, dict):
            out.update(_flatten_report(
                v, prefix=key,
                include_list_lengths=include_list_lengths,
                include_numeric_list_stats=include_numeric_list_stats,
                include_boolean_allowlist=()  # subdicts: keep it simple
            ))

        elif isinstance(v, list):
            if include_list_lengths:
                out[f"{key}__len"] = float(len(v))
            if v and include_numeric_list_stats and all(_is_number(x) for x in v):
                arr = np.array(v, dtype=float)
                out[f"{key}__mean"] = float(arr.mean())
                out[f"{key}__min"]  = float(arr.min())
                out[f"{key}__max"]  = float(arr.max())
        # ignore strings/None/etc
    return out

@dataclass
class FeatureConfig:
    allowlist: Iterable[str] = field(default_factory=list)
    denylist: Iterable[str] = field(default_factory=list)

    def filter(self, feats: Dict[str, float]) -> Dict[str, float]:
        keys = list(feats.keys())
        if self.allowlist:
            al = set(self.allowlist)
            keys = [k for k in keys if k in al]
        if self.denylist:
            dl = set(self.denylist)
            keys = [k for k in keys if k not in dl]
        return {k: feats[k] for k in keys}

# ---------- IO ----------

def load_examples_from_folder(data_dir: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    Reads every JSON file in `data_dir` that matches `pattern`.
    Each file must contain a JSON array of items with fields: q, a, c (list[str]), y (0/1).
    """
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    items: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            try:
                arr = json.load(f)
                if not isinstance(arr, list):
                    raise ValueError(f"{p} does not contain a JSON array.")
                for i, ex in enumerate(arr):
                    # minimal validation
                    if not {"q","a","c","y"}.issubset(ex):
                        raise ValueError(f"{p}[{i}] missing required keys.")
                    if not isinstance(ex["c"], list):
                        raise ValueError(f"{p}[{i}] 'c' must be a list of strings.")
                    items.append(ex)
            except Exception as e:
                raise RuntimeError(f"Failed to parse {p}: {e}")
    return items

# ---------- Featurization ----------

def featurize_item(
    ex: Dict[str, Any],
    metrics_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], int, Dict[str, Any]]:
    """
    Build report → flatten → return (flat_features, label, full_report) for a single item.
    """
    rep = context_utilization_report_with_entities(
        question=ex["q"],
        answer=ex["a"],
        retrieved_contexts=ex["c"],
        use_bm25_for_best=True,
        use_embed_alignment=False,
        metrics_config=metrics_config,
    )
    flat = _flatten_report(
        rep,
        include_list_lengths=True,
        include_numeric_list_stats=True,
        include_boolean_allowlist=("inference_likely",),
    )
    y = int(ex["y"])
    return flat, y, rep

def featurize_dataset(
    data: List[Dict[str, Any]],
    metrics_config: Optional[Dict[str, Any]] = None,
    feature_config: Optional[FeatureConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Returns:
      X: (N, D) float array
      y: (N,) int array
      feature_names: list[str] (columns of X)
      meta: list of dicts with "q","a","y" (handy for auditing)
    """
    flat_list: List[Dict[str, float]] = []
    y_list: List[int] = []
    meta: List[Dict[str, Any]] = []

    for ex in data:
        flat, y, _rep = featurize_item(ex, metrics_config=metrics_config)
        if feature_config:
            flat = feature_config.filter(flat)
        flat_list.append(flat)
        y_list.append(y)
        meta.append({"q": ex["q"], "a": ex["a"], "y": y})

    # union of all feature keys -> consistent matrix
    all_keys = sorted(set().union(*[f.keys() for f in flat_list])) if flat_list else []
    X = np.zeros((len(flat_list), len(all_keys)), dtype=float)
    for i, f in enumerate(flat_list):
        for j, k in enumerate(all_keys):
            if k in f:
                X[i, j] = f[k]

    y = np.array(y_list, dtype=int)
    return X, y, all_keys, meta

# ---------- Save ----------

def save_csv(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], meta: List[Dict[str, Any]], out_csv: str
) -> None:
    """
    Saves a single CSV with columns: feature_names..., y, q, a
    (meta strings at the end for convenience when auditing.)
    """
    # Write CSV manually to avoid pandas dependency if you prefer
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        header = feature_names + ["y", "q", "a"]
        f.write(",".join([json.dumps(h)[1:-1] if ("," in h or "." in h) else h for h in header]) + "\n")
        for i in range(X.shape[0]):
            row_vals: List[str] = []
            for j in range(X.shape[1]):
                row_vals.append(f"{X[i, j]:.6f}")
            row_vals.append(str(int(y[i])))
            # escape q,a as JSON then strip outer quotes so commas are safe
            qi = json.dumps(meta[i]["q"])
            ai = json.dumps(meta[i]["a"])
            row_vals.append(qi)
            row_vals.append(ai)
            f.write(",".join(row_vals) + "\n")

def save_npz(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], out_npz: str
) -> None:
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, X=X, y=y, feature_names=np.array(feature_names, dtype=object))

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Folder with *.json files (each a list of items)")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob to match files in data-dir")
    ap.add_argument("--out-prefix", type=str, default="processed/run1", help="Output prefix (no extension)")
    # optional metric toggles you already support in metric_utils
    ap.add_argument("--enable-pos-metrics", action="store_true")
    ap.add_argument("--enable-inference-signal", action="store_true")
    # optional simple include/exclude for features
    ap.add_argument("--allow", nargs="*", default=[], help="Exact feature names to keep")
    ap.add_argument("--deny", nargs="*", default=[], help="Exact feature names to drop")
    args = ap.parse_args()

    metrics_config = {
        "enable_pos_metrics": bool(args.enable_pos_metrics),
        "enable_inference_signal": bool(args.enable_inference_signal),
    }

    # 1) Load
    data = load_examples_from_folder(args.data_dir, args.pattern)
    if not data:
        raise SystemExit(f"No examples found in {args.data_dir}/{args.pattern}")

    # 2) Featurize
    feat_cfg = FeatureConfig(allowlist=args.allow, denylist=args.deny)
    X, y, names, meta = featurize_dataset(data, metrics_config=metrics_config, feature_config=feat_cfg)

    # 3) Save
    out_csv = f"{args.out_prefix}.csv"
    out_npz = f"{args.out_prefix}.npz"
    save_csv(X, y, names, meta, out_csv)
    save_npz(X, y, names, out_npz)

    print(f"✅ Saved {X.shape[0]} rows, {X.shape[1]} features")
    print(f"   CSV: {out_csv}")
    print(f"   NPZ: {out_npz}")

if __name__ == "__main__":
    main()