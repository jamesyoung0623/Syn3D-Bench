#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${INTERNVL3_MODEL_NAME:-OpenGVLab/InternVL3_5-2B-Instruct}"
MODEL_SIZE="${INTERNVL3_MODEL_SIZE:-2B}"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
MAX_VIDEOS="${INTERNVL3_MAX_VIDEOS:-100}"
CHECKPOINT_ROOT="${INTERNVL3_CHECKPOINT_ROOT:-${SCRIPT_DIR}/internvl3_finetune/checkpoints}"
EXCLUDE_OBJECT_IDS_PATH="${INTERNVL3_EXCLUDE_OBJECT_IDS_PATH:-${SCRIPT_DIR}/internvl3_finetune/train_reason_replay_object_ids.txt}"

LORA_RANKS=(${INTERNVL3_LORA_RANKS:-4 8 16 32})
CHECKPOINTS=(${INTERNVL3_CHECKPOINTS:-224 252 final})

if [[ -f "${EXCLUDE_OBJECT_IDS_PATH}" ]]; then
  export INTERNVL3_EXCLUDE_OBJECT_IDS_PATH="${EXCLUDE_OBJECT_IDS_PATH}"
  echo "exclude_object_ids_path=${EXCLUDE_OBJECT_IDS_PATH}"
elif [[ -n "${INTERNVL3_EXCLUDE_OBJECT_IDS_PATH:-}" ]]; then
  echo "Missing INTERNVL3_EXCLUDE_OBJECT_IDS_PATH: ${EXCLUDE_OBJECT_IDS_PATH}" >&2
  exit 1
fi

for lora_r in "${LORA_RANKS[@]}"; do
  run_tag="internvl3_${MODEL_SIZE_SLUG}_lora_${lora_r}_SI_consensus_uncertain_reason_sft"
  output_base="internvl3_${MODEL_SIZE_SLUG}_lora_${lora_r}_SI_consensus_uncertain"

  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ "${checkpoint}" == "final" ]]; then
      checkpoint_dir="checkpoint-final"
      output_tag="${output_base}"
    else
      checkpoint_dir="checkpoint-${checkpoint}"
      output_tag="${output_base}_${checkpoint}"
    fi

    echo "=== InternVL3 inference: ${run_tag}/${checkpoint_dir} ==="
    INTERNVL3_MODEL_NAME="${MODEL_NAME}" \
    INTERNVL3_MODEL_SIZE="${MODEL_SIZE}" \
    INTERNVL3_FINETUNE_DIR="${CHECKPOINT_ROOT}/${run_tag}/${checkpoint_dir}" \
    INTERNVL3_OUTPUT_TAG="${output_tag}" \
    INTERNVL3_MAX_VIDEOS="${MAX_VIDEOS}" \
    python inference_InternVL3.py
  done
done
