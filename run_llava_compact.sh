#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL_SIZE="${LLAVA_MODEL_SIZE:-4B}"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
MAX_VIDEOS="${LLAVA_MAX_VIDEOS:-100}"
MAX_NEW_TOKENS="${LLAVA_MAX_NEW_TOKENS:-128}"
NUM_SAMPLED_FRAMES="${LLAVA_NUM_SAMPLED_FRAMES:-12}"
CHECKPOINT_ROOT="${LLAVA_CHECKPOINT_ROOT:-${SCRIPT_DIR}/llava_onevision_finetune/checkpoints}"
EXCLUDE_OBJECT_IDS_PATH="${LLAVA_EXCLUDE_OBJECT_IDS_PATH:-${SCRIPT_DIR}/llava_onevision_finetune/train_exclude_object_ids.txt}"

case "${MODEL_SIZE_SLUG}" in
  4b)
    MODEL_NAME="${LLAVA_MODEL_NAME:-lmms-lab/LLaVA-OneVision-1.5-4B-Instruct}"
    RUN_TAG_PREFIX="${LLAVA_RUN_TAG_PREFIX:-llava_onevision_lora}"
    OUTPUT_PREFIX="${LLAVA_OUTPUT_PREFIX:-llava_onevision_4b_lora}"
    ;;
  8b)
    MODEL_NAME="${LLAVA_MODEL_NAME:-lmms-lab/LLaVA-OneVision-1.5-8B-Instruct}"
    RUN_TAG_PREFIX="${LLAVA_RUN_TAG_PREFIX:-llava_onevision_8b_lora}"
    OUTPUT_PREFIX="${LLAVA_OUTPUT_PREFIX:-llava_onevision_8b_lora}"
    ;;
  *)
    echo "Unsupported LLAVA_MODEL_SIZE=${MODEL_SIZE}. Use 4B or 8B." >&2
    exit 2
    ;;
esac

LORA_RANKS=(${LLAVA_LORA_RANKS:-4 8 16 32})
CHECKPOINTS=(${LLAVA_CHECKPOINTS:-224 252 final})

if [[ -f "${EXCLUDE_OBJECT_IDS_PATH}" ]]; then
  export LLAVA_EXCLUDE_OBJECT_IDS_PATH="${EXCLUDE_OBJECT_IDS_PATH}"
  echo "exclude_object_ids_path=${EXCLUDE_OBJECT_IDS_PATH}"
elif [[ -n "${LLAVA_EXCLUDE_OBJECT_IDS_PATH:-}" ]]; then
  echo "Missing LLAVA_EXCLUDE_OBJECT_IDS_PATH: ${EXCLUDE_OBJECT_IDS_PATH}" >&2
  exit 1
fi

for lora_r in "${LORA_RANKS[@]}"; do
  run_tag="${RUN_TAG_PREFIX}_${lora_r}_ST_consensus_uncertain_reason_sft"
  output_base="${OUTPUT_PREFIX}_${lora_r}_ST_consensus_uncertain"

  for checkpoint in "${CHECKPOINTS[@]}"; do
    if [[ "${checkpoint}" == "final" ]]; then
      checkpoint_dir="checkpoint-final"
      output_tag="${output_base}"
    else
      checkpoint_dir="checkpoint-${checkpoint}"
      output_tag="${output_base}_${checkpoint}"
    fi

    echo "=== LLaVA-OneVision inference: ${run_tag}/${checkpoint_dir} ==="
    LLAVA_MODEL_NAME="${MODEL_NAME}" \
    LLAVA_MODEL_SIZE="${MODEL_SIZE}" \
    LLAVA_FINETUNE_DIR="${CHECKPOINT_ROOT}/${run_tag}/${checkpoint_dir}" \
    LLAVA_OUTPUT_TAG="${output_tag}" \
    LLAVA_MAX_VIDEOS="${MAX_VIDEOS}" \
    LLAVA_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    LLAVA_NUM_SAMPLED_FRAMES="${NUM_SAMPLED_FRAMES}" \
    python inference_LLAVA.py
  done
done
