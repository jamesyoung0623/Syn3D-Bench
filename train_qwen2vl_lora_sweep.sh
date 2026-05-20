#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_IDS="${QWEN2VL_CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${QWEN2VL_NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS
MODEL_SIZE="2B"
MODEL_SIZE_SLUG="${MODEL_SIZE,,}"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_NUM_SAMPLED_FRAMES=6
DEFAULT_FRAME_MAX_PIXELS=200704
TRAIN_JSONL="${QWEN2VL_TRAIN_JSONL:-internvl3_finetune/all_projects_label_only_SI_consensus_uncertain.jsonl}"
TRAIN_JSONL_BASENAME="$(basename "${TRAIN_JSONL}")"
DATASET_TAG="${TRAIN_JSONL_BASENAME#all_projects_label_only_}"
DATASET_TAG="${DATASET_TAG#train_full_}"
DATASET_TAG="${DATASET_TAG%.jsonl}"
DATASET_TAG="${DATASET_TAG%_consensus_uncertain}"
OUTPUT_ROOT="${QWEN2VL_OUTPUT_ROOT:-qwen2vl_finetune/checkpoints}"
OUTPUT_PREFIX="${QWEN2VL_OUTPUT_PREFIX:-qwen2vl_${MODEL_SIZE_SLUG}_lora}"
PER_DEVICE_BATCH="${QWEN2VL_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRAD_ACCUM="${QWEN2VL_GRADIENT_ACCUMULATION_STEPS:-2}"
EPOCHS="${QWEN2VL_NUM_TRAIN_EPOCHS:-5}"
SAVE_TOTAL_LIMIT="${QWEN2VL_SAVE_TOTAL_LIMIT:-5}"
MAX_LENGTH="${QWEN2VL_MAX_LENGTH:-4096}"
NUM_SAMPLED_FRAMES="${QWEN2VL_NUM_SAMPLED_FRAMES:-${DEFAULT_NUM_SAMPLED_FRAMES}}"
FRAME_MAX_PIXELS="${QWEN2VL_FRAME_MAX_PIXELS:-${DEFAULT_FRAME_MAX_PIXELS}}"
LORA_ALPHA="${QWEN2VL_LORA_ALPHA:-32}"
LLM_LR="${QWEN2VL_LLM_LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${QWEN2VL_WARMUP_STEPS:-28}"
LORA_RANKS=(${QWEN2VL_LORA_RANKS:-4 8 16 32})

for lora_r in "${LORA_RANKS[@]}"; do
  output_dir="${OUTPUT_ROOT}/${OUTPUT_PREFIX}_${lora_r}_${DATASET_TAG}_consensus_uncertain_reason_sft"

  echo "=== Qwen2-VL ${MODEL_SIZE} SFT: lora_r=${lora_r} ==="
  echo "model_name=${MODEL_NAME}"
  echo "train_jsonl=${TRAIN_JSONL}"
  echo "output_dir=${output_dir}"
  echo "num_sampled_frames=${NUM_SAMPLED_FRAMES}"
  echo "frame_max_pixels=${FRAME_MAX_PIXELS}"
  echo "max_length=${MAX_LENGTH}"
  train_args=(
    train_qwen2vl_sft.py
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
    --llm_lora True \
    --lora_r "${lora_r}" \
    --lora_alpha "${LORA_ALPHA}" \
    --llm_learning_rate "${LLM_LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --num_sampled_frames "${NUM_SAMPLED_FRAMES}" \
    --frame_max_pixels "${FRAME_MAX_PIXELS}"
  )

  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${train_args[@]}"
  else
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" CUDA_VISIBLE_DEVICES="${GPU_IDS}" python "${train_args[@]}"
  fi
done
