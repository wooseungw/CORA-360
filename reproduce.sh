#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="$ROOT/release/model_registry.json"
ACTION="${1:-help}"

lookup() {
  python - "$REGISTRY" "$1" "$2" <<'PY'
import json, sys
registry, model_id, key = sys.argv[1:]
models = json.load(open(registry, encoding="utf-8"))["models"]
model = next((item for item in models if item["id"] == model_id), None)
if model is None:
    raise SystemExit(f"Unknown model id: {model_id}")
print(model[key])
PY
}

prepare_config() {
  local model_id="$1" source_config="$2" output_config="$3" output_dir="$4"
  local base_model base_revision snapshot
  base_model="$(lookup "$model_id" base_model)"
  base_revision="$(lookup "$model_id" base_revision)"
  snapshot="$(python - "$base_model" "$base_revision" <<'PY'
from huggingface_hub import snapshot_download
import sys
print(snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2]))
PY
)"
  python - "$source_config" "$output_config" "$snapshot" "$output_dir" <<'PY'
import sys, yaml
source, output, snapshot, output_dir = sys.argv[1:]
config = yaml.safe_load(open(source, encoding="utf-8"))
config["model"]["hf_model_id"] = snapshot
config["model"]["processor_id"] = snapshot
config["output_dir"] = output_dir
yaml.safe_dump(config, open(output, "w", encoding="utf-8"), sort_keys=False)
PY
}

case "$ACTION" in
  data-check)
    python "$ROOT/scripts/verify_data_manifest.py" --train "$2" --test "$3"
    ;;
  evaluate)
    MODEL_ID="$2"
    TEST_CSV="$3"
    CONFIG="$(lookup "$MODEL_ID" config)"
    HF_REPO="$(lookup "$MODEL_ID" hf_repo)"
    OUTPUT_DIR="$ROOT/reproductions/$MODEL_ID"
    mkdir -p "$OUTPUT_DIR"
    prepare_config "$MODEL_ID" "$ROOT/$CONFIG" "$OUTPUT_DIR/config.yaml" "$OUTPUT_DIR"
    MODEL_NAME="$(python - "$OUTPUT_DIR/config.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["model"]["name"])
PY
)"
    ADAPTER_DIR="$OUTPUT_DIR/$MODEL_NAME/lora_adapter"
    mkdir -p "$ADAPTER_DIR" "$OUTPUT_DIR"
    python - "$HF_REPO" "$ADAPTER_DIR" <<'PY'
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
    python "$ROOT/scripts/baseline_eval.py" --config "$OUTPUT_DIR/config.yaml" --test-csv "$TEST_CSV" \
      --train-output-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/eval"
    ;;
  verify)
    python "$ROOT/scripts/verify_reproduction.py" --model "$2" --actual "$3"
    ;;
  train)
    MODEL_ID="$2"
    TRAIN_CSV="$3"
    TEST_CSV="$4"
    CONFIG="$(lookup "$MODEL_ID" config)"
    OUTPUT_DIR="$ROOT/reproductions/$MODEL_ID-train"
    mkdir -p "$OUTPUT_DIR"
    prepare_config "$MODEL_ID" "$ROOT/$CONFIG" "$OUTPUT_DIR/config.yaml" "$OUTPUT_DIR"
    python - "$OUTPUT_DIR/config.yaml" "$TRAIN_CSV" "$TEST_CSV" <<'PY'
import sys, yaml
source, train, test = sys.argv[1:]
config = yaml.safe_load(open(source, encoding="utf-8"))
config["data_train_csv"] = train
config["data_test_csv"] = test
yaml.safe_dump(config, open(source, "w", encoding="utf-8"), sort_keys=False)
PY
    python "$ROOT/scripts/baseline_finetune.py" --config "$OUTPUT_DIR/config.yaml"
    ;;
  *)
    echo "Usage: $0 {data-check TRAIN TEST|evaluate MODEL TEST|verify MODEL METRICS|train MODEL TRAIN TEST}"
    exit 2
    ;;
esac
