#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_NAME="${QWEN2VL_MODEL_NAME:-Qwen/Qwen2-VL-2B-Instruct}"
MODEL_SIZE="${QWEN2VL_MODEL_SIZE:-2B}"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
MAX_VIDEOS="${QWEN2VL_MAX_VIDEOS:-100}"
MAX_NEW_TOKENS="${QWEN2VL_MAX_NEW_TOKENS:-128}"
NUM_SAMPLED_FRAMES="${QWEN2VL_NUM_SAMPLED_FRAMES:-12}"
FRAME_MAX_PIXELS="${QWEN2VL_FRAME_MAX_PIXELS:-200704}"
CHECKPOINT_ROOT="${QWEN2VL_CHECKPOINT_ROOT:-${SCRIPT_DIR}/qwen2vl_finetune/checkpoints}"
EXCLUDE_OBJECT_IDS_PATH="${QWEN2VL_EXCLUDE_OBJECT_IDS_PATH:-${SCRIPT_DIR}/qwen2vl_finetune/train_exclude_object_ids.txt}"

LORA_RANKS=(${QWEN2VL_LORA_RANKS:-4 8 16 32})
CHECKPOINTS=(${QWEN2VL_CHECKPOINTS:-224 252 final})

if [[ -f "${EXCLUDE_OBJECT_IDS_PATH}" ]]; then
  export QWEN2VL_EXCLUDE_OBJECT_IDS_PATH="${EXCLUDE_OBJECT_IDS_PATH}"
  echo "exclude_object_ids_path=${EXCLUDE_OBJECT_IDS_PATH}"
elif [[ -n "${QWEN2VL_EXCLUDE_OBJECT_IDS_PATH:-}" ]]; then
  echo "Missing QWEN2VL_EXCLUDE_OBJECT_IDS_PATH: ${EXCLUDE_OBJECT_IDS_PATH}" >&2
  exit 1
fi

for lora_r in "${LORA_RANKS[@]}"; do
  run_tag="qwen2vl_${MODEL_SIZE_SLUG}_lora_${lora_r}_ST_consensus_uncertain_reason_sft"
  output_base="qwen2vl_${MODEL_SIZE_SLUG}_lora_${lora_r}_ST_consensus_uncertain"

  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ "${checkpoint}" == "final" ]]; then
      checkpoint_dir="checkpoint-final"
      output_tag="${output_base}"
    else
      checkpoint_dir="checkpoint-${checkpoint}"
      output_tag="${output_base}_${checkpoint}"
    fi

    echo "=== Qwen2-VL inference: ${run_tag}/${checkpoint_dir} ==="
    QWEN2VL_MODEL_NAME="${MODEL_NAME}" \
    QWEN2VL_MODEL_SIZE="${MODEL_SIZE}" \
    QWEN2VL_FINETUNE_DIR="${CHECKPOINT_ROOT}/${run_tag}/${checkpoint_dir}" \
    QWEN2VL_OUTPUT_TAG="${output_tag}" \
    QWEN2VL_MAX_VIDEOS="${MAX_VIDEOS}" \
    QWEN2VL_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    QWEN2VL_NUM_SAMPLED_FRAMES="${NUM_SAMPLED_FRAMES}" \
    QWEN2VL_FRAME_MAX_PIXELS="${FRAME_MAX_PIXELS}" \
    python inference_Qwen2-VL.py
  done
done
