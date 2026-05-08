#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_NAME="${INTERNVL3_MODEL_NAME:-}"
MODEL_SIZE="${INTERNVL3_MODEL_SIZE:-}"
RUN_TAG="${INTERNVL3_RUN_TAG:-}"
if [[ -z "${MODEL_NAME}" || -z "${MODEL_SIZE}" || -z "${RUN_TAG}" ]]; then
  echo "Set INTERNVL3_MODEL_NAME, INTERNVL3_MODEL_SIZE, and INTERNVL3_RUN_TAG before running." >&2
  echo "Example:" >&2
  echo "  INTERNVL3_MODEL_NAME=OpenGVLab/InternVL3_5-4B-Instruct \\" >&2
  echo "  INTERNVL3_MODEL_SIZE=4B \\" >&2
  echo "  INTERNVL3_RUN_TAG=internvl3_5_4b \\" >&2
  echo "  $0 all" >&2
  exit 2
fi
FINETUNE_DIR="${INTERNVL3_FINETUNE_ROOT:-${SCRIPT_DIR}/internvl3_finetune}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_internvl3_sft.py"
INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_InternVL3.py"
GPU_IDS="${INTERNVL3_CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${INTERNVL3_NPROC_PER_NODE:-2}"

COMBOS=(SI SL SS ST)

run_train() {
  for combo in "${COMBOS[@]}"; do
    echo
    echo "=== InternVL3 SFT train: ${combo} ==="
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "${TRAIN_SCRIPT}" \
      --model_name "${MODEL_NAME}" \
      --train_jsonl "${FINETUNE_DIR}/train_full_${combo}.jsonl" \
      --output_dir "${FINETUNE_DIR}/checkpoints/${RUN_TAG}_${combo}_sft" \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --num_train_epochs 6 \
      --logging_steps 1 \
      --save_strategy epoch \
      --save_total_limit 2 \
      --bf16 True \
      --gradient_checkpointing True \
      --tune_projector True \
      --lora_r 8 \
      --projector_learning_rate 5e-5 \
      --warmup_ratio 0.1 \
      --llm_learning_rate 2e-5
  done
}

run_infer() {
  for combo in "${COMBOS[@]}"; do
    echo
    echo "=== InternVL3 inference: ${combo} ==="
    INTERNVL3_MODEL_NAME="${MODEL_NAME}" \
    INTERNVL3_MODEL_SIZE="${MODEL_SIZE}" \
    INTERNVL3_FINETUNE_DIR="${FINETUNE_DIR}/checkpoints/${RUN_TAG}_${combo}_sft/checkpoint-final" \
    python "${INFERENCE_SCRIPT}"
  done
}

case "${MODE}" in
  train)
    run_train
    ;;
  infer|inference)
    run_infer
    ;;
  all)
    run_train
    run_infer
    ;;
  *)
    echo "Usage: $0 [train|infer|all]" >&2
    exit 2
    ;;
esac
