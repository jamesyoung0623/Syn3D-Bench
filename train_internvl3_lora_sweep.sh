#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_IDS="${INTERNVL3_CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${INTERNVL3_NPROC_PER_NODE:-2}"
MODEL_SIZE="${INTERNVL3_MODEL_SIZE:-2B}"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
case "${MODEL_SIZE_SLUG}" in
  1b)
    DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-1B-Instruct"
    ;;
  2b)
    DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-2B-Instruct"
    ;;
  4b)
    DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-4B-Instruct"
    ;;
  8b)
    DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-8B-Instruct"
    ;;
  14b)
    DEFAULT_MODEL_NAME="OpenGVLab/InternVL3_5-14B-Instruct"
    ;;
  *)
    echo "Unsupported INTERNVL3_MODEL_SIZE=${MODEL_SIZE}. Use 1B, 2B, 4B, 8B, or 14B." >&2
    exit 2
    ;;
esac
MODEL_NAME="${INTERNVL3_MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
TRAIN_JSONL="${INTERNVL3_TRAIN_JSONL:-internvl3_finetune/train_full_SI_consensus_uncertain.jsonl}"
OUTPUT_ROOT="${INTERNVL3_OUTPUT_ROOT:-internvl3_finetune/checkpoints}"
OUTPUT_PREFIX="${INTERNVL3_OUTPUT_PREFIX:-internvl3_${MODEL_SIZE_SLUG}_lora}"
PER_DEVICE_BATCH="${INTERNVL3_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${INTERNVL3_GRADIENT_ACCUMULATION_STEPS:-2}"
EPOCHS="${INTERNVL3_NUM_TRAIN_EPOCHS:-10}"
SAVE_TOTAL_LIMIT="${INTERNVL3_SAVE_TOTAL_LIMIT:-3}"
PROJECTOR_LR="${INTERNVL3_PROJECTOR_LEARNING_RATE:-5e-5}"
LLM_LR="${INTERNVL3_LLM_LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${INTERNVL3_WARMUP_RATIO:-0.1}"
LORA_ALPHA="${INTERNVL3_LORA_ALPHA:-32}"
LORA_RANKS=(${INTERNVL3_LORA_RANKS:-4 8 16 32})

for lora_r in "${LORA_RANKS[@]}"; do
  output_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}_${lora_r}_SI_consensus_uncertain_reason_sft"

  echo "=== InternVL3 ${MODEL_SIZE} SFT: lora_r=${lora_r} ==="
  echo "model_name=${MODEL_NAME}"
  echo "output_dir=${output_dir}"

  train_args=(
    train_internvl3_sft.py
    --model_name "${MODEL_NAME}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${output_dir}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_train_epochs "${EPOCHS}" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bf16 True \
    --gradient_checkpointing True \
    --tune_projector True \
    --llm_lora True \
    --lora_r "${lora_r}" \
    --lora_alpha "${LORA_ALPHA}" \
    --projector_learning_rate "${PROJECTOR_LR}" \
    --llm_learning_rate "${LLM_LR}" \
    --warmup_ratio "${WARMUP_RATIO}"
  )

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${train_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python "${train_args[@]}"
  fi
done
