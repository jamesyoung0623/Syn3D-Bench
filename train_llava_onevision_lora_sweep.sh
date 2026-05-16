#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_IDS="${LLAVA_CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${LLAVA_NPROC_PER_NODE:-2}"
HF_HUB_OFFLINE="${LLAVA_HF_HUB_OFFLINE:-${HF_HUB_OFFLINE:-0}}"
MODEL_SIZE="${LLAVA_MODEL_SIZE:-4B}"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
case "${MODEL_SIZE_SLUG}" in
  4b)
    DEFAULT_MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    DEFAULT_OUTPUT_PREFIX="llava_onevision_lora"
    ;;
  8b)
    DEFAULT_MODEL_NAME="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    DEFAULT_OUTPUT_PREFIX="llava_onevision_8b_lora"
    ;;
  *)
    echo "Unsupported LLAVA_MODEL_SIZE=${MODEL_SIZE}. Use 4B or 8B." >&2
    exit 2
    ;;
esac
MODEL_NAME="${LLAVA_MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
TRAIN_JSONL="${LLAVA_TRAIN_JSONL:-internvl3_finetune/train_full_ST_consensus_uncertain.jsonl}"
OUTPUT_ROOT="${LLAVA_OUTPUT_ROOT:-llava_onevision_finetune/checkpoints}"
OUTPUT_PREFIX="${LLAVA_OUTPUT_PREFIX:-${DEFAULT_OUTPUT_PREFIX}}"
MAX_LENGTH="${LLAVA_MAX_LENGTH:-8192}"
PER_DEVICE_BATCH="${LLAVA_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${LLAVA_GRADIENT_ACCUMULATION_STEPS:-2}"
EPOCHS="${LLAVA_NUM_TRAIN_EPOCHS:-10}"
SAVE_TOTAL_LIMIT="${LLAVA_SAVE_TOTAL_LIMIT:-3}"
PROJECTOR_LR="${LLAVA_PROJECTOR_LEARNING_RATE:-5e-5}"
LLM_LR="${LLAVA_LLM_LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${LLAVA_WARMUP_RATIO:-0.1}"
LORA_ALPHA="${LLAVA_LORA_ALPHA:-32}"
LORA_RANKS=(${LLAVA_LORA_RANKS:-4 8 16 32})

for lora_r in "${LORA_RANKS[@]}"; do
  output_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}_${lora_r}_ST_consensus_uncertain_reason_sft"

  echo "=== LLaVA-OneVision ${MODEL_SIZE} SFT: lora_r=${lora_r} ==="
  echo "model_name=${MODEL_NAME}"
  echo "output_dir=${output_dir}"

  train_args=(
    train_llava_onevision_sft.py
    --model_size "${MODEL_SIZE}" \
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
    --warmup_ratio "${WARMUP_RATIO}" \
    --max_length "${MAX_LENGTH}"
  )

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${train_args[@]}"
  else
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" CUDA_VISIBLE_DEVICES="${GPU_IDS}" python "${train_args[@]}"
  fi
done
