# data_processing.py
from __future__ import annotations
import os, glob, json, argparse
from typing import Any, Dict, List, Iterable, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Use your metrics report
import logging

from metric_utils import context_utilization_report_with_entities
from shared_config import add_extractor_flags, metrics_config_from_args

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
    logger = logging.getLogger(__name__)
    try:
        rep = context_utilization_report_with_entities(
            question=ex.get("q", ""),
            answer=ex.get("a", ""),
            retrieved_contexts=ex.get("c", []) or [],
            use_bm25_for_best=getattr(metrics_config, "use_bm25_for_best", True),
            use_embed_alignment=getattr(metrics_config, "use_embed_alignment", True),
            extractor_config=getattr(metrics_config, "extractor", None),
            timezone=getattr(metrics_config, "timezone", "UTC"),
            metrics_config=metrics_config,
        )
    except Exception:
        logger.exception("context_utilization_report_with_entities failed for question=%r", ex.get("q", ""))
        raise
    flat = _flatten_report(
        rep,
        include_list_lengths=True,
        include_numeric_list_stats=True,
        include_boolean_allowlist=("inference_likely",),
    )
    enable_entity_report = True
    if metrics_config is not None:
        try:
            if isinstance(metrics_config, dict):
                enable_entity_report = bool(metrics_config.get("enable_entity_report", True))
            else:
                enable_entity_report = bool(getattr(metrics_config, "enable_entity_report", True))
        except Exception:
            enable_entity_report = True

    if enable_entity_report:
        # Expose per-type supported entity counts as flat features (e.g., entity_match.DATE__len)
        se_by_type = ((rep.get("supported_entities") or {}).get("by_type") or {})
        for t, cnt in se_by_type.items():
            flat[f"entity_match.{t}__len"] = float(cnt)
        # Ensure canonical types exist even if zero
        for t in ("MONEY", "NUMBER", "PERCENT", "DATE", "QUANTITY", "PHONE"):
            flat.setdefault(f"entity_match.{t}__len", 0.0)
    else:
        # Strip entity-derived features when reporting is disabled
        entity_prefixes = ("entity_match.", "supported_entities.")
        flat = {
            k: v
            for k, v in flat.items()
            if not k.startswith(entity_prefixes) and k not in {"unsupported_entity_count"}
        }
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

    logger = logging.getLogger(__name__)
    skipped: List[Tuple[int, str]] = []
    for idx, ex in enumerate(data):
        try:
            flat, y, _rep = featurize_item(ex, metrics_config=metrics_config)
        except Exception:
            logger.exception("Failed to featurize item index=%s question=%r; skipping", idx, ex.get("q", ""))
            skipped.append((idx, ex.get("q", "")))
            continue
        if feature_config:
            flat = feature_config.filter(flat)
        flat_list.append(flat)
        y_list.append(y)
        meta.append({"q": ex["q"], "a": ex["a"], "y": y})
        if "product launch event" in (ex.get("q", "") or "").lower():
            logger.info(
                "Debug entity stats for question=%r: answer=%r DATE_len=%s presence=%s supported=%s extractor_date=%s spacy_fusion=%s rep=%s",
                ex.get("q", ""),
                ex.get("a", ""),
                flat.get("entity_match.DATE__len"),
                flat.get("entity_match.presence_by_type.DATE"),
                ((_rep.get("supported_entities") or {}).get("by_type") if isinstance(_rep, dict) else None),
                getattr(metrics_config, "extractor", None).enable_date if getattr(metrics_config, "extractor", None) else None,
                getattr(metrics_config, "extractor", None).use_spacy_fusion if getattr(metrics_config, "extractor", None) else None,
                _rep if isinstance(_rep, dict) else None,
            )

    if skipped:
        logger.warning(
            "Skipped %d items due to featurization errors. First few: %s",
            len(skipped),
            [f"(idx={i}, q={q!r})" for i, q in skipped[:3]],
        )

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
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data", help="Folder with *.json files (each a list of items)")
    ap.add_argument("--pattern", type=str, default="*.json", help="Glob to match files in data-dir")
    ap.add_argument("--out-prefix", type=str, default="processed/run1", help="Output prefix (no extension)")
    # optional simple include/exclude for features
    ap.add_argument("--allow", nargs="*", default=[], help="Exact feature names to keep")
    ap.add_argument("--deny", nargs="*", default=[], help="Exact feature names to drop")
    add_extractor_flags(ap)
    args = ap.parse_args()
    logger.info("Parsed args: %s", vars(args))

    metrics_config = metrics_config_from_args(args)

    extractor_cfg = getattr(metrics_config, "extractor", None)
    try:
        import spacy  # noqa: F401
        spacy_available = True
    except Exception:
        spacy_available = False

    logging.getLogger("extractor").setLevel(logging.WARNING)

    logger.info(
        "Configuration: use_spacy_fusion=%s, enable_entity_report=%s, spacy_available=%s",
        getattr(extractor_cfg, "use_spacy_fusion", None),
        getattr(metrics_config, "enable_entity_report", None),
        spacy_available,
    )
    if extractor_cfg is not None:
        logger.info(
        "Extractor enable flags: DATE=%s MONEY=%s NUMBER=%s PERCENT=%s use_spacy_for=%s",
        extractor_cfg.enable_date,
        extractor_cfg.enable_money,
        extractor_cfg.enable_number,
        extractor_cfg.enable_percent,
        extractor_cfg.use_spacy_for,
        )

    # 1) Load
    data = load_examples_from_folder(args.data_dir, args.pattern)
    if not data:
        raise SystemExit(f"No examples found in {args.data_dir}/{args.pattern}")
    logger.info("Loaded %d examples from %s/%s", len(data), args.data_dir, args.pattern)

    extractor_cfg_for_test = getattr(metrics_config, "extractor", None)
    if extractor_cfg_for_test is not None:
        for ex in data:
            if "product launch event" in (ex.get("q", "") or "").lower():
                try:
                    import extractor as _extractor_mod
                    _extract_entities = _extractor_mod.extract_entities
                    _ensure_spacy = _extractor_mod._ensure_spacy
                    _extract_spacy = _extractor_mod._extract_spacy

                    ents = _extract_entities(ex.get("a", ""), config=extractor_cfg_for_test)
                    logger.info(
                        "Pre-flight extract_entities for question=%r -> %s (spacy_loaded=%s)",
                        ex.get("q", ""),
                        [(e.type, e.source) for e in ents],
                        _extractor_mod._SPACY_NLP is not None,
                    )
                    if _extractor_mod._SPACY_NLP is not None:
                        logger.info(
                            "spaCy pipeline components: %s",
                            _extractor_mod._SPACY_NLP.pipe_names,
                        )
                    if True:
                        spacy_only = _extract_spacy(ex.get("a", ""), extractor_cfg_for_test)
                        logger.info("_extract_spacy direct -> %s", [(e.type, e.source) for e in spacy_only])
                    if _extractor_mod._SPACY_NLP is None:
                        ensured = _ensure_spacy()
                        logger.warning(
                            "_ensure_spacy() returned: %s (after call loaded=%s)",
                            ensured,
                            _extractor_mod._SPACY_NLP is not None,
                        )
                        ents_retry = _extract_entities(ex.get("a", ""), config=extractor_cfg_for_test)
                        logger.warning(
                            "Retry extract_entities -> %s",
                            [(e.type, e.source) for e in ents_retry],
                        )
                except Exception:
                    logger.exception("Pre-flight extract_entities failed")
                break

    # 2) Featurize
    feat_cfg = FeatureConfig(allowlist=args.allow, denylist=args.deny)
    X, y, names, meta = featurize_dataset(data, metrics_config=metrics_config, feature_config=feat_cfg)

    def _count(feature_name: str) -> int:
        if feature_name not in names:
            return 0
        col = names.index(feature_name)
        return int(np.count_nonzero(X[:, col]))

    logger.info(
        "Non-zero counts: DATE_len=%d, MONEY_len=%d, NUMBER_len=%d",
        _count("entity_match.DATE__len"),
        _count("entity_match.MONEY__len"),
        _count("entity_match.NUMBER__len"),
    )
    if "supported_entities.by_type.DATE" in names:
        logger.info(
            "Non-zero supported DATE rows=%d",
            _count("supported_entities.by_type.DATE"),
        )
    else:
        logger.warning("supported_entities.by_type.DATE column missing from feature set")

    # Log a sample row where DATE is present for diagnostics
    if "entity_match.DATE__len" in names:
        col = names.index("entity_match.DATE__len")
        for idx, row in enumerate(X):
            if row[col] > 0:
                logger.info(
                    "Sample DATE row idx=%d question=%r DATE_len=%.3f",
                    idx,
                    meta[idx]["q"],
                    row[col],
                )
                break
        else:
            logger.warning("No DATE matches found in dataset")

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
