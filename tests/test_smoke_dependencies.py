"""Smoke tests that ensure critical optional dependencies load correctly."""


def test_spacy_version():
    import spacy

    nlp = spacy.blank("en")
    version = spacy.__version__
    assert version.startswith("3."), f"Unexpected spaCy version: {version}"
    assert hasattr(nlp, "pipe_names") or hasattr(nlp, "pipeline"), "spaCy pipeline API missing"


def test_sentence_transformers():
    import os
    import pytest
    from sentence_transformers import SentenceTransformer

    model_path = None
    for env_var in ("RAG_MODELS_DIR", "MODELS_DIR"):
        base = os.environ.get(env_var)
        if base:
            candidate = os.path.join(base, "all-MiniLM-L6-v2")
            if os.path.exists(candidate):
                model_path = candidate
                break

    if model_path is None:
        pytest.skip("SentenceTransformer model cache not available; set RAG_MODELS_DIR for full smoke test")

    model = SentenceTransformer(model_path)
    emb = model.encode(["smoke test"], normalize_embeddings=True)
    assert emb.shape[0] == 1


def test_dateparser_roundtrip():
    import dateparser

    dt = dateparser.parse("September 20, 2023", settings={"TIMEZONE": "UTC"})
    assert dt is not None and dt.year == 2023
