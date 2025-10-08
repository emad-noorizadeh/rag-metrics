#!/usr/bin/env bash
set -euo pipefail

# Edit the defaults below and re-run the script when you need a different config.
PYTHON_BIN="python"
DATA_DIR="data"
PATTERN="*.json"
OUT_PREFIX="data/processed/train_v6"
ALLOW_FEATURES=""
# Drop redundant/duplicate signals identified by scripts/analyze_features.py for lr_model_v1.
DENY_FEATURES="
supported_entities.by_type.MONEY
supported_entities.by_type.NUMBER
supported_entities.by_type.PERCENT
supported_entities.by_type.QUANTITY
unsupported_entity_count
unsupported_mass
hallucination_risk_score
min_sentence_precision
p90_sentence_precision
per_sentence__max
per_sentence__mean
per_sentence__min
precision_token
"
ENABLE_POS_METRICS="true"
ENABLE_INFERENCE_SIGNAL="true"
ENABLE_ENTITY_REPORT="true"
USE_SPACY_FUSION="true"
PREFER_DETERMINISTIC="auto"
TIMEZONE="UTC"
ENABLE_TYPES=""
SPACY_FOR=""
SOURCE_WEIGHT_DET=""
SOURCE_WEIGHT_SPACY=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

lower() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# Optional positional overrides: first arg can be a file or directory.
if [[ $# -gt 0 && ${1:0:1} != '-' ]]; then
    input_path=$1
    shift
    if [[ -f "$input_path" ]]; then
        DATA_DIR=$(dirname "$input_path")
        PATTERN=$(basename "$input_path")
    elif [[ -d "$input_path" ]]; then
        DATA_DIR="$input_path"
        if [[ $# -gt 0 && ${1:0:1} != '-' ]]; then
            PATTERN=$1
            shift
        fi
    else
        echo "create_dataset.sh: '$input_path' is not a file or directory" >&2
        exit 1
    fi
fi

cmd=("$PYTHON_BIN" "data_processing.py" "--data-dir" "$DATA_DIR" "--pattern" "$PATTERN" "--out-prefix" "$OUT_PREFIX")

declare -a _allow=()
declare -a _deny=()

if [[ -n "${ALLOW_FEATURES//[[:space:]]/}" ]]; then
    IFS=$' \t\n' read -r -a _allow <<< "$ALLOW_FEATURES"
fi

if [[ -n "${DENY_FEATURES//[[:space:]]/}" ]]; then
    IFS=$' \t\n' read -r -a _deny <<< "$DENY_FEATURES"
fi

if ((${#_allow[@]})); then
    cmd+=("--allow")
    cmd+=("${_allow[@]}")
fi

if ((${#_deny[@]})); then
    cmd+=("--deny")
    cmd+=("${_deny[@]}")
fi

if [[ $(lower "$ENABLE_POS_METRICS") == "true" ]]; then
    cmd+=("--enable-pos-metrics")
fi

if [[ $(lower "$ENABLE_INFERENCE_SIGNAL") == "true" ]]; then
    cmd+=("--enable-inference-signal")
fi

if [[ $(lower "$ENABLE_ENTITY_REPORT") == "true" ]]; then
    cmd+=("--enable-entity-report")
fi

case $(lower "$USE_SPACY_FUSION") in
    true) cmd+=("--use-spacy-fusion") ;;
    *) ;;
esac

case $(lower "$PREFER_DETERMINISTIC") in
    true) cmd+=("--prefer-deterministic") ;;
    false) cmd+=("--no-prefer-deterministic") ;;
    *) ;;
esac

if [[ -n "${TIMEZONE// }" ]]; then
    cmd+=("--timezone" "$TIMEZONE")
fi

if [[ -n "${ENABLE_TYPES// }" ]]; then
    read -r -a _types <<< "$ENABLE_TYPES"
    cmd+=("--enable-types")
    cmd+=("${_types[@]}")
fi

if [[ -n "${SPACY_FOR// }" ]]; then
    read -r -a _spacy_for <<< "$SPACY_FOR"
    cmd+=("--spacy-for")
    cmd+=("${_spacy_for[@]}")
fi

if [[ -n "${SOURCE_WEIGHT_DET// }" ]]; then
    cmd+=("--source-weight-det" "$SOURCE_WEIGHT_DET")
fi

if [[ -n "${SOURCE_WEIGHT_SPACY// }" ]]; then
    cmd+=("--source-weight-spacy" "$SOURCE_WEIGHT_SPACY")
fi

# Forward any additional CLI arguments to data_processing.py.
if [[ $# -gt 0 ]]; then
    cmd+=("$@")
fi

echo "Running data_processing with:" >&2
printf '  %q' "${cmd[@]}" >&2
printf '\n' >&2

"${cmd[@]}"
