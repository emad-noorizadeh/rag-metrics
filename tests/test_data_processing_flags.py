# tests/test_data_processing_flags.py
import types
from data_processing import featurize_item
from shared_config import metrics_config_from_args

def _mk_args(**overrides):
    # mimic argparse.Namespace the way shared_config expects
    ns = types.SimpleNamespace(
        enable_pos_metrics=False,
        enable_inference_signal=False,
        enable_entity_report=True,
        use_spacy_fusion=False,   # default off
        timezone="UTC",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns

def test_relative_date_requires_fusion():
    ex = {
        "q": "When is it?",
        "a": "next Friday",
        "c": ["We meet next Friday."],
        "y": 1,
    }

    # no fusion
    args = _mk_args(use_spacy_fusion=False)
    cfg = metrics_config_from_args(args)
    flat_no, y_no, rep_no = featurize_item(ex, metrics_config=cfg)
    # Expect fewer/nonexistent DATE cues
    assert flat_no.get("entity_match.DATE__len", 0.0) <= 0.0

    # with fusion
    args = _mk_args(use_spacy_fusion=True)
    cfg = metrics_config_from_args(args)
    flat_yes, y_yes, rep_yes = featurize_item(ex, metrics_config=cfg)
    # Expect DATE features picked up now
    assert flat_yes.get("entity_match.DATE__len", 0.0) >= 1.0
# tests/test_data_processing_flags.py
import types
from data_processing import featurize_item
from shared_config import metrics_config_from_args


def _mk_args(**overrides):
    """Mimic argparse.Namespace the way shared_config expects."""
    ns = types.SimpleNamespace(
        enable_pos_metrics=False,
        enable_inference_signal=False,
        enable_entity_report=True,
        use_spacy_fusion=False,   # default off
        timezone="UTC",
        # allow extractor feature toggles to be passed through
        enable_number=True,
        enable_money=True,
        enable_date=True,
        enable_quantity=True,
        enable_percent=True,
        enable_phone=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_entity_report_toggle_respected():
    """When enable_entity_report is False, no entity_match.* features should be produced.
    When True, entity features should be present for matching inputs.
    """
    ex = {
        "q": "When is it?",
        "a": "2024-07-24",
        "c": ["We meet on 2024-07-24."],
        "y": 1,
    }

    # entity report disabled
    args = _mk_args(enable_entity_report=False)
    cfg = metrics_config_from_args(args)
    flat_off, y_off, rep_off = featurize_item(ex, metrics_config=cfg)
    assert not any(k.startswith("entity_match.") for k in flat_off.keys())

    # entity report enabled
    args = _mk_args(enable_entity_report=True)
    cfg = metrics_config_from_args(args)
    flat_on, y_on, rep_on = featurize_item(ex, metrics_config=cfg)
    assert any(k.startswith("entity_match.") for k in flat_on.keys())


def test_extractor_flag_enable_number_controls_numbers():
    """The extractor's enable_number flag should govern NUMBER extraction in features."""
    ex = {
        "q": "How many?",
        "a": "three",
        "c": ["We saw three."],
        "y": 1,
    }

    # disable NUMBER extraction
    args = _mk_args(enable_entity_report=True, enable_number=False)
    cfg = metrics_config_from_args(args)
    flat_off, y_off, rep_off = featurize_item(ex, metrics_config=cfg)
    assert flat_off.get("entity_match.NUMBER__len", 0.0) == 0.0

    # enable NUMBER extraction
    args = _mk_args(enable_entity_report=True, enable_number=True)
    cfg = metrics_config_from_args(args)
    flat_on, y_on, rep_on = featurize_item(ex, metrics_config=cfg)
    assert flat_on.get("entity_match.NUMBER__len", 0.0) >= 1.0


def test_config_threads_use_spacy_fusion_flag():
    """Sanity check: the shared config threads the fusion flag into the extractor config."""
    args = _mk_args(use_spacy_fusion=False)
    cfg = metrics_config_from_args(args)
    assert getattr(cfg.extractor, "use_spacy_fusion", False) is False

    args = _mk_args(use_spacy_fusion=True)
    cfg = metrics_config_from_args(args)
    assert getattr(cfg.extractor, "use_spacy_fusion", False) is True