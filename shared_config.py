# shared_config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable
import argparse
from types import SimpleNamespace
from extractor import ExtractorConfig

# ---- Public API -----------------------------------------------------------------

def add_extractor_flags(parser: argparse.ArgumentParser) -> None:
    """
    Attach a consistent set of flags to any argparse parser.
    Use this in CLIs that ultimately call metric_utils.context_utilization_report_with_entities.
    """
    # High-level metric toggles
    parser.add_argument("--enable-pos-metrics", action="store_true",
                        help="Include part-of-speech based metrics in report (if supported).")
    parser.add_argument("--enable-inference-signal", action="store_true",
                        help="Include heuristic 'inference likely' signal (if supported).")
    parser.add_argument("--enable-entity-report", action="store_true",
                        help="Emit entity_match features in flattened outputs.")

    # Extractor fusion & behavior
    parser.add_argument("--use-spacy-fusion", action="store_true",
                        help="Enable spaCy-backed fusion (keeps deterministic patterns and fuses).")
    parser.add_argument("--prefer-deterministic", dest="prefer_deterministic",
                        action="store_true", default=None,
                        help="Prefer deterministic candidates on ties.")
    parser.add_argument("--no-prefer-deterministic", dest="prefer_deterministic",
                        action="store_false",
                        help="Disable deterministic tie-preference.")
    parser.add_argument("--timezone", type=str, default=None,
                        help="Timezone used for date normalization (e.g., 'UTC').")

    # Type gating / per-type spaCy
    ALL_TYPES = ["MONEY", "DATE", "QUANTITY", "PHONE", "NUMBER", "PERCENT"]
    parser.add_argument("--enable-types", nargs="*", default=None, choices=ALL_TYPES,
                        help="Whitelist of entity types to extract. If omitted, use extractor defaults.")
    parser.add_argument("--spacy-for", nargs="*", default=None, choices=ALL_TYPES,
                        help="When using spaCy fusion, enable spaCy candidates only for these types.")

    # Source weighting
    parser.add_argument("--source-weight-det", type=float, default=None,
                        help="Scoring weight for deterministic candidates.")
    parser.add_argument("--source-weight-spacy", type=float, default=None,
                        help="Scoring weight for spaCy candidates.")

def metrics_config_from_args(args: argparse.Namespace) -> SimpleNamespace:
    """
    Convert parsed args → SimpleNamespace consumed by data_processing/metrics_utils.
    Provides a ready-to-use ExtractorConfig at `.extractor` and mirrors other toggles.
    """
    # Build ExtractorConfig from args (falling back to sensible defaults)
    # Map spacy_for list (if provided) into the dict shape ExtractorConfig expects.
    spacy_for_map = None
    if getattr(args, "spacy_for", None):
        spacy_for_map = {t: True for t in args.spacy_for}

    # Map source weights if provided.
    source_weights = {}
    if getattr(args, "source_weight_det", None) is not None:
        source_weights["det"] = float(args.source_weight_det)
    if getattr(args, "source_weight_spacy", None) is not None:
        source_weights["spacy"] = float(args.source_weight_spacy)
    if not source_weights:
        source_weights = None

    def _flag(name: str, default: bool) -> bool:
        val = getattr(args, name, None)
        if val is None:
            return default
        return bool(val)

    timezone = getattr(args, "timezone", None) or "UTC"

    extractor_cfg = ExtractorConfig(
        enable_money=_flag("enable_money", True),
        enable_date=_flag("enable_date", True),
        enable_quantity=_flag("enable_quantity", True),
        enable_phone=_flag("enable_phone", False),
        enable_number=_flag("enable_number", True),
        enable_percent=_flag("enable_percent", True),
        # fusion & behavior
        timezone=timezone,
        use_spacy_fusion=getattr(args, "use_spacy_fusion", False),
        prefer_deterministic=getattr(args, "prefer_deterministic", None),
        use_spacy_for=spacy_for_map,
        source_weights=source_weights,
    )

    # Optional coarse type gating via enable_types list
    enable_types_list = getattr(args, "enable_types", None)
    if enable_types_list:
        allow = set(enable_types_list)
        extractor_cfg.enable_money = extractor_cfg.enable_money and ("MONEY" in allow)
        extractor_cfg.enable_date = extractor_cfg.enable_date and ("DATE" in allow)
        extractor_cfg.enable_quantity = extractor_cfg.enable_quantity and ("QUANTITY" in allow)
        extractor_cfg.enable_phone = extractor_cfg.enable_phone and ("PHONE" in allow)
        extractor_cfg.enable_number = extractor_cfg.enable_number and ("NUMBER" in allow)
        extractor_cfg.enable_percent = extractor_cfg.enable_percent and ("PERCENT" in allow)

    # Compose a compact namespace for downstream code.
    enable_types_raw = getattr(args, "enable_types", None)
    enable_types_list = list(enable_types_raw) if enable_types_raw else None

    return SimpleNamespace(
        # metric toggles
        enable_pos_metrics=getattr(args, "enable_pos_metrics", False),
        enable_inference_signal=getattr(args, "enable_inference_signal", False),
        enable_entity_report=getattr(args, "enable_entity_report", False),
        # shared extractor config
        extractor=extractor_cfg,
        # convenience mirror of timezone for features that read it directly
        timezone=timezone,
        # keep type gating at the top level as well if other callers rely on it
        enable_types=enable_types_list,
        # pass-through for advanced consumers; most will rely on extractor fields above
        use_spacy_for=spacy_for_map,
        source_weights=source_weights,
    )

# ---- Optional: typed container if you want a dataclass in code -------------------

@dataclass
class MetricsConfig:
    # metrics toggles
    enable_pos_metrics: bool = False
    enable_inference_signal: bool = False

    # extractor / fusion
    use_spacy_fusion: bool = False
    prefer_deterministic: Optional[bool] = None
    timezone: Optional[str] = None

    # type gating
    enable_types: Optional[List[str]] = None
    use_spacy_for: Optional[Dict[str, bool]] = None

    # source weights
    source_weights: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """
        Emit the compact dict expected by metric_utils/context_utilization_report_with_entities.
        Excludes fields set to their defaults/None when appropriate.
        """
        d = {}
        if self.enable_pos_metrics: d["enable_pos_metrics"] = True
        if self.enable_inference_signal: d["enable_inference_signal"] = True
        if self.use_spacy_fusion: d["use_spacy_fusion"] = True
        if self.prefer_deterministic is not None:
            d["prefer_deterministic"] = bool(self.prefer_deterministic)
        if self.timezone: d["timezone"] = self.timezone
        if self.enable_types: d["enable_types"] = list(self.enable_types)
        if self.use_spacy_for: d["use_spacy_for"] = dict(self.use_spacy_for)
        if self.source_weights: d["source_weights"] = dict(self.source_weights)
        return d
