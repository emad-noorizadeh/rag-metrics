# RAG Metrics Pipeline

This project turns retrieval-augmented QA traces into structured metrics and
supervised labels that drive a lightweight classifier for groundedness. The
system is built around three stages:

```
┌────────────┐   ┌──────────────────────────┐   ┌────────────────────┐
│ Raw Inputs │──▶│ context_utilization_     │──▶│ Flatten & Feature   │
│ (q, a, c)  │   │ report_with_entities()   │   │ Vector (numpy/CSV)  │
└────────────┘   └──────────────────────────┘   └────────────────────┘
                        ▲            │                    │
                        │            └── uses ────────────┘
                        │
                        └── Extractor (regex + spaCy fusion)
```

The flattened feature matrix feeds the `eval_kfold.py` training script during
modeling and the `predict_batch.py` / `run_full_report.py` utilities during
inference.

## Architecture Overview

### Environment Setup

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. Download the MiniLM checkpoint and point the loader at it. Either:

   ```bash
   mkdir -p /opt/models
   git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
     /opt/models/all-MiniLM-L6-v2
   export RAG_MODELS_DIR=/opt/models
   ```

   or copy the extracted folder somewhere convenient and set
   `RAG_MODELS_DIR` (or `MODELS_DIR`) to that directory in your shell / service
   environment. `_maybe_load_embedder` will look there first, then fall back to
   `../models/all-MiniLM-L6-v2/` or `./models/all-MiniLM-L6-v2/` relative to the
   repo if no env variable is set.

### 1. Entity Extraction (`extractor.py`)

The extractor merges deterministic pattern matchers with optional spaCy NER to
produce typed entities:

| Type      | Deterministic logic                                     | spaCy fusion
|-----------|---------------------------------------------------------|-------------
| `MONEY`   | Regex + currency map, numeric suffixes (k/m/b)          | `en_core_web_sm` MONEY → normalized via regex
| `NUMBER`  | Regex for numerals & abbreviations, `number_parser`     | spaCy CARDINAL/ORDINAL (optional)
| `PERCENT` | Regex for `%` tokens                                    | spaCy PERCENT
| `DATE`    | `dateparser` w/ relative phrases & ISO fast path        | spaCy DATE (with day backfill for “July 2024”)
| `QUANTITY`| Regex with canonical unit aliases                       | spaCy QUANTITY (optional)
| `PHONE`   | `phonenumbers` (optional)                               | spaCy PHONE (optional)

Key knobs (see `shared_config.py:add_extractor_flags`):

* `--use-spacy-fusion` – include spaCy candidates.
* `--enable-<TYPE>` – enable/disable deterministic sources per type.
* `--spacy-for TYPE …` – enable spaCy per type when fusion is on.
* `--timezone` – forwarded to `dateparser` after the fix (defaults to `UTC`).

During data processing the script performs a “pre-flight” extraction log for a
sentinel question and reports skipped examples if any featurization fails.

### 2. Context Utilization Report (`metric_utils.py`)

`context_utilization_report_with_entities()` combines lexical, numeric, and
entity alignment metrics. Core outputs:

* **Lexical precision/recall** – `precision_token`, `recall_context`, per-sentence
  stats, TF-IDF similarities.
* **Semantic alignment** – optional minilm embeddings for Q↔A and A↔Context.
* **Numeric alignment** – `numeric_match`, `unsupported_numbers`, etc.
* **Entity alignment** – `entity_match` dictionary (overall rate plus
  per-type `presence_by_type` and `state_by_type`), `supported_entities` with
  spans, and `unsupported` lists.
* **POS-aware features** – when `--enable-pos-metrics` is set, the report
  adds `content_precision_token`, `content_term_support_rate`, etc.
* **Unsupported hallmarks** – unsupported term counts, top-k impact, and
  numeric/entity hallucination indicators.

All metrics are deterministic given the extractor configuration.

### 3. Feature Flattening (`data_processing.py`)

`featurize_item()` flattens the nested report into scalar features, preserving
list lengths, numeric aggregates, and boolean flags (think `foo.bar__len` or
`qr_alignment.cosine_tfidf`). Entity metrics receive additional convenience
columns (e.g., `entity_match.DATE__len`).

`featurize_dataset()` applies this to every example, optionally filtering via
`FeatureConfig(allowlist=…, denylist=…)`. Failures are logged and skipped so a
single bad record does not abort the run.

CLI usage:

```
/Users/emadn/Projects/pipven/bin/python data_processing.py \
  --data-dir data --pattern "*.json" \
  --out-prefix data/processed/train_v1 \
  --use-spacy-fusion --enable-pos-metrics --enable-entity-report
```

The script logs key diagnostics:

* Parsed flag values (ensure spaCy fusion & entity report toggles match).
* SpaCy availability and pipeline components.
* Sample entity counts, including the number of rows with DATE matches.

The generated artifacts are `train_v1.csv` (human-inspectable) and
`train_v1.npz` (numpy arrays with feature names).

## Feature Space Cheat Sheet

Features are grouped naturally by their prefix:

* **Lexical Coverage** – `precision_token`, `recall_context`, `qa_len_ratio`,
  `per_sentence__*`, `unsupported_term_*`.
* **Entity Metrics** –
  * Per-type counts: `entity_match.<TYPE>__len`.
  * Presence flags: `entity_match.presence_by_type.<TYPE>` (1 if answer mentions).
  * Match state: `entity_match.state_by_type.<TYPE>` (1 fully grounded, 0 partial,
    -1 absent in answer).
  * Aggregates: `entity_match.overall`, `entity_match.unsupported__len`.
  * Supported spans: `supported_entities.by_type.<TYPE>`, `supported_entities.count`.
* **Numeric Alignment** – `numeric_match`, `unsupported_numbers__len`,
  `unsupported_numeric_rate`, `unsupported_entity_mass`.
* **Context Similarity** – `context_alignment.best_context_similarity`,
  `context_alignment.answer_context_similarity`.
* **Question–Answer Alignment** – `qr_alignment.*` (TF-IDF and optional embedders).
* **POS / Head Noun Stats** – `content_precision_token`, `head_noun_support_rate`.
* **Quick-win heuristics** – `hallucination_risk_score`, `best_context_len` and
  related coverage proxies.

The final feature count depends on the toggles you pass (entity reporting adds
27 columns, POS metrics add four, etc.).

## Data Pipeline and Safety Nets

1. **Extraction sanity check** – before featurization, the script calls
   `extract_entities` for a sentinel question and logs the spaCy pipeline so a
   missing model is obvious.
2. **Featurization** – errors are caught per example; failing items are skipped
   with a warning summary.
3. **Artifact emission** – both CSV/NPZ outputs include row order, question, and
   answer text for traceability.

## Training (`eval_kfold.py`)

`eval_kfold.py` consumes the `.npz` files and trains a logistic regression using
stratified k-fold cross-validation:

```
/Users/emadn/Projects/pipven/bin/python eval_kfold.py \
  --train-npz data/processed/train_v3.npz \
  --test-npz  data/processed/test_v3.npz \
  --test-csv  data/processed/test_v3.csv \
  --Cs 0.1,0.3,1,3,10 --n-splits 5 --objective f1 \
  --min-precision 0.90 --standardize --max-iter 2000 \
  --solver lbfgs --penalty l2 --class-weight balanced \
  --use-embed-alignment --seed 42 \
  --deny unsupported_entity_count unsupported_mass \
  --save-model artifacts/lr_model_v1.pkl \
  --save-report artifacts/cv_report_v1.json \
  --featurization-meta artifacts/featurization_meta.json \
  --tag "v1.0-trainset_v3"
```

Passing `--test-csv` in tandem with `--test-npz` lets the script join the
`answer_type` column from the CSV and print a per-category breakdown. Use the
new `--allow`/`--deny` flags to keep or drop specific feature columns *after*
loading the NPZ. The same mask is applied to the optional test NPZ so train/test
stay aligned.

Feature filtering notes:

- `--deny unsupported_entity_count unsupported_mass` drops those columns while
  leaving the rest untouched.
- `--allow context_alignment.best_context_similarity numeric_match` keeps only
  the listed feature names (and implicitly drops everything else).
- Any filters applied are recorded in both `cv_report*.json` and the saved
  model payload as `feature_filter` / `feature_config`. The provided
  `featurization_meta.json` is also augmented with a
  `post_filter_feature_config` block so inference code can reapply the exact
  mask when reconstructing feature vectors.

Outputs:

* `artifacts/lr_model_v1.pkl` – pickled scikit-learn model + metadata.
* `artifacts/cv_report_v1.json` – per-fold metrics, thresholds per C, test-set
  evaluation.
* `artifacts/featurization_meta.json` – captures feature names, upstream
  extractor toggles, and (if you used `--allow/--deny`) a
  `post_filter_feature_config` describing which columns were removed. Load this
  alongside the model to ensure runtime featurization uses the identical
  configuration.

### Feature reference & enabling flags

The featurizer always produces the core lexical coverage signals and then
optionally adds richer metrics based on the CLI flags you pass to
`data_processing.py` (or to `create_dataset.sh`, which wraps the same flags).
The table below summarises the main feature families, the columns they expand
to, and how to turn them on.

| Feature family | Example flattened columns | What it captures | Enable via |
| --- | --- | --- | --- |
| **Lexical coverage (default)** | `precision_token`, `recall_context`, `best_context_share`, `per_sentence__{len,mean,min,max}`, `unsupported_term_rate`, `unsupported_mass` | Token overlap between answer and retrieved context, including per-sentence precision and unsupported token counts | Always on (can be disabled by editing `metrics_config` keys) |
| **Context statistics (default)** | `best_context_len`, `num_contexts`, `avg_ctx_len_tokens`, `supported_topk_impact`, `unsupported_to_supported_ratio` | Quick diagnostics about retrieval depth and the balance of supported vs unsupported terms | Always on |
| **Numeric grounding** | `numeric_match`, `unsupported_numeric_rate`, `unsupported_numbers__len` | Agreement between answer numbers and the retrieved numeric evidence | Always on |
| **Entity grounding** | `entity_match.MONEY__len`, `supported_entities.by_type.DATE`, `unsupported_entity_count`, `entity_match.overall` | Typed entity alignment between answer and context; per-type counts and overall ratio | `--enable-entity-report` (default off) |
| **POS-aware support** | `content_precision_token`, `content_term_support_rate`, `content_unsupported_mass`, `head_noun_support_rate` | Overlap restricted to content words / head nouns using spaCy POS tags | `--enable-pos-metrics` (requires spaCy) |
| **Inference detector** | `inference_score`, `inference_likely_bool` | Heuristic that fires when answers look inferred (low lexical match, high typed match, decent semantic similarity) | `--enable-inference-signal` |
| **QA lexical alignment** | `qr_alignment.cosine_tfidf`, `qr_alignment.answer_covers_question` | TF-IDF similarity between the question and answer and how completely the answer covers the question terms | Always on |
| **QA semantic alignment** | `qr_alignment.cosine_embed`, `qr_alignment.answer_covers_question_sem` | MiniLM embedding similarity between question and answer | Enabled automatically when `sentence-transformers` MiniLM model is available (record `--use-embed-alignment` for bookkeeping) |
| **Answer↔context semantic alignment** | `context_alignment.answer_context_similarity`, `context_alignment.best_context_similarity` | Embedding similarity between the answer and each retrieved context chunk | Same as above (requires MiniLM embeddings) |
| **Per-type entity lengths** | `entity_match.DATE__len`, `entity_match.PERCENT__len`, etc. | Number of supported entities per type in the answer text | Automatically included when `--enable-entity-report` is set |
| **Boolean allowlist** | `inference_likely_bool`, `inference_likely` | Booleans flattened to 0/1 when allowlisted | Allowlist managed by `data_processing.py` (defaults include `inference_likely`) |

Lists (e.g., unsupported term details) are flattened with suffixes like
`__len`, `__mean`, `__min`, `__max`. If you need to gate individual metrics you
can extend `metrics_config` in `shared_config.py` or pass overrides like
`--prefer-deterministic`/`--enable-types` to control the extractor.

### Feature diagnostics (`analyze_features.py`)

Use `scripts/analyze_features.py` to inspect the featurized dataset and spot
redundant columns, zero-variance signals, or dominant logistic coefficients.

Example:

```
python scripts/analyze_features.py \
  --npz data/processed/train_v1.npz \
  --model artifacts/lr_model_v1.pkl \
  --out artifacts/feature_report_v1.json
```

Parameters:

- `--npz` (required): path to the NPZ produced by `data_processing.py`. It must
  include `X`, `y`, and `feature_names` arrays.
- `--model` (optional): pickled artifact from `eval_kfold.py`; when supplied the
  script extracts logistic-regression coefficients via `_extract_lr`.
- `--out` (optional): JSON path to persist the diagnostics (same content that is
  printed to stdout).
- `--corr-threshold` (optional, default 0.9): minimum absolute Pearson
  correlation when flagging highly correlated feature pairs.

What you get:

- List of zero-variance columns (`zero_variance_features`).
- Top feature correlations with the label (`top_corr_with_y`).
- Highly correlated feature pairs (`high_correlation_pairs`) to guide deny
  lists.
- Top positive / negative logistic coefficients (`top_positive_coefficients`,
  `top_negative_coefficients`) when a model is provided.
- Optional JSON report suitable for checking into `artifacts/` alongside other
  diagnostics.

Tip: feed the flagged features into `DENY_FEATURES` in `create_dataset.sh` or
pass them to `eval_kfold.py --deny` so training and inference stay aligned after
pruning.

Recent run highlights (train_v3/test_v3):

| Metric | CV (C=3.0) | Test |
|--------|------------|------|
| F1     | 0.577      | 0.931|
| Precision | 0.748  | 0.904|
| Recall | 0.586      | 0.959|

Top contributing features showed the expected importance shifts once DATE
features were re-enabled (e.g., `entity_match.presence_by_type.DATE` flipped to
positive weight, DATE counts penalize misses).

## Runtime Inference

Use the same featurization pipeline and the saved artifacts:

1. Featurize raw inputs with the same flags used during training (reuse the
   command above or the API inside your service).
2. Load `featurization_meta.json` and re-align any feature vectors (e.g., drop
   unseen columns or insert zeros for missing ones).
3. Load `lr_model_v1.pkl`, apply the learned threshold (`cv_report_v1.json`
   records the recommended value), and emit groundedness predictions alongside
   probability scores.

`predict_batch.py` shows a minimal example of this flow; `run_full_report.py`
drives a richer JSON summary that can include per-example metrics for error
analysis.

### Single-Example Workflow

To score a single raw QA trace end-to-end:

1. **Prepare JSON input** containing a list of examples (even for one item):

   ```json
   [
     {
       "q": "When was the product launch event held?",
       "a": "The product launch event was on September 20, 2023.",
       "c": [
         "Event Summary: The product was officially launched on September 20, 2023, with a live demonstration.",
         "Press Release: The launch attracted significant media attention and customer interest."
       ]
     }
   ]
   ```

   Save this as `samples/single.json`.

2. **Featurize** with the same flags used during training:

   ```bash
/Users/emadn/Projects/pipven/bin/python data_processing.py \
  --data-dir samples --pattern "single.json" \
  --out-prefix artifacts/single_run \
  --use-spacy-fusion --enable-pos-metrics --enable-entity-report
   ```

   This emits `artifacts/single_run.csv` (for inspection) and
   `artifacts/single_run.npz` (for inference). Any featurization errors are
   logged and the offending items skipped.

3. **Score** with the trained model:

   ```bash
   /Users/emadn/Projects/pipven/bin/python scripts/predict_batch.py \
     --model artifacts/lr_model_v1.pkl \
     --npz artifacts/single_run.npz \
     --out artifacts/single_run_scores.jsonl
   ```

   The JSONL output reports the probability (`p`) and binary prediction
   (`yhat`) for each row, along with the original question/answer text. The
   threshold used comes from the saved classifier artifact.

If you need to mix source types (e.g., JSON plus the CSV fixture in
`tests/data/sample_inputs.csv`), pass a comma-separated glob:

```
/Users/emadn/Projects/pipven/bin/python data_processing.py \
  --data-dir tests/data --pattern "*.json,*.csv" \
  --out-prefix artifacts/train_combo \
  --use-spacy-fusion --enable-pos-metrics --enable-entity-report
```

For a pure metric dump (without classification), use
`demo/run_full_report.py --input demo/example.json --pretty` on a single example
formatted as shown above; it calls the report directly without flattening.

## Keeping Train & Inference in Sync

* **Flag parity** – always supply the same extractor/metric flags to
  `data_processing.py` and to your runtime featurization. `featurization_meta.json`
  encodes the expected feature list; treat it as part of the model artifact.
* **Timezone default** – `shared_config.metrics_config_from_args()` now defaults
  to `'UTC'` when `--timezone` is omitted so spaCy DATE parsing is deterministic.
* **Skipped rows** – errors are skipped at featurization time and reported. Keep
  an eye on the warning count; a non-zero value suggests malformed inputs.
* **Embedding models** – `_maybe_load_embedder` checks (in order) the path passed
  in code, environment overrides `RAG_MODELS_DIR`/`MODELS_DIR`, and a repo-local
  `../models/…` folder. Configure `RAG_MODELS_DIR` in production to point at the
  shared checkpoint cache (e.g., `/opt/models`).

## Additional Notes

* The repo still contains `classifier.py` with the toy dataset for ad-hoc
  experiments. The production pipeline uses the CSV/NPZ outputs instead.
* Logging is intentionally verbose around entity extraction; if you need quieter
  runs, lower the log level or remove the sentinel debug block.
* Tests under `tests/` cover extractor edge cases (`test_entities_hardcases`),
  flag plumbing (`test_data_processing_flags`), and end-to-end metrics sanity.
* Run `scripts/analyze_features.py --npz ... --model ...` to inspect feature
  correlations, zero-variance columns, and coefficient rankings after training.

---

For a quick sanity check after regenerating features:

```
rg "entity_match.DATE__len" data/processed/train_v1.csv | head
```

You should see non-zero values for rows that contain grounded dates. That, plus
the logger’s “Non-zero counts” summary, confirms the spaCy fusion path is live.
