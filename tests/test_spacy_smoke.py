"""Minimal spaCy smoke test."""

import warnings


warnings.filterwarnings(
    "ignore",
    message=".*split_arg_string is deprecated.*",
    category=DeprecationWarning,
)


def test_spacy_smoke():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("The product launch event was on September 20, 2023.")
    ents = {(ent.text, ent.label_) for ent in doc.ents}
    assert ("September 20, 2023", "DATE") in ents
    assert hasattr(nlp, "pipe_names") or hasattr(nlp, "pipeline")
