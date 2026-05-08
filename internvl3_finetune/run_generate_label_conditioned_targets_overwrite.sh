#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$DATA_DIR/generate_label_conditioned_targets.py"
MODEL_NAME="OpenGVLab/InternVL3-8B"
IMAGE_SIZE=448
NUM_SAMPLED_FRAMES=12

run_job() {
  local split="$1"
  local input_jsonl="$DATA_DIR/all_projects_label_only_${split}.jsonl"
  local output_jsonl="$DATA_DIR/train_full_${split}.jsonl"

  echo "[$(date -u +%FT%TZ)] Starting ${split}"
  python -u "$SCRIPT_PATH" \
    --input_jsonl "$input_jsonl" \
    --output_jsonl "$output_jsonl" \
    --model_name "$MODEL_NAME" \
    --image_size "$IMAGE_SIZE" \
    --num_sampled_frames "$NUM_SAMPLED_FRAMES" \
    --overwrite
  echo "[$(date -u +%FT%TZ)] Finished ${split}"
}

run_job SI
run_job SL
run_job SS
run_job ST

echo "[$(date -u +%FT%TZ)] All jobs finished"
