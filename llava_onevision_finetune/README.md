**LLaVA-OneVision Workspace**

This folder mirrors `qwen2vl_finetune` for LLaVA-OneVision LoRA SFT runs.
It wraps `../train_llava_onevision_lora_sweep.sh` and stores adapters under
`checkpoints/`.

**Training**

Default training uses the 4B model, two GPUs, and LoRA ranks `4 8 16 32`:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
llava_onevision_finetune/run_llava_onevision_sft.sh train
```

Useful overrides:

```bash
LLAVA_MODEL_SIZE=8B \
LLAVA_MODEL_NAME=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
LLAVA_LORA_RANKS="4 8 16 32" \
LLAVA_MAX_LENGTH=8192 \
LLAVA_CUDA_VISIBLE_DEVICES=0,1 \
LLAVA_NPROC_PER_NODE=2 \
llava_onevision_finetune/run_llava_onevision_sft.sh train
```

By default checkpoints are written to:

```text
llava_onevision_finetune/checkpoints/llava_onevision_lora_${rank}_ST_consensus_uncertain_reason_sft
```

**Inference**

Run the compact LLaVA inference sweep over the folder checkpoints. By default,
this excludes objects used in training:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
llava_onevision_finetune/run_llava_onevision_inference.sh
```

Default inference settings:
- model size: `4B`
- LoRA ranks: `4 8 16 32`
- checkpoints: `224 252 final`
- max videos per project: `100`
- sampled frames per video: `12`
- max output tokens: `128`

Use only final checkpoints:

```bash
LLAVA_CHECKPOINTS="final" llava_onevision_finetune/run_llava_onevision_inference.sh
```

**Exclude Training Objects**

`infer`, `inference`, and the no-argument default all skip training objects:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
llava_onevision_finetune/run_llava_onevision_inference.sh infer
```

This writes `train_exclude_object_ids.txt` and sets
`LLAVA_EXCLUDE_OBJECT_IDS_PATH` before running inference. It looks for
`train_reason_replay_*.jsonl` and `train_full_*.jsonl` in this folder first,
then falls back to `internvl3_finetune`.

To include all objects without exclusion:

```bash
llava_onevision_finetune/run_llava_onevision_inference.sh infer-all
```

**Useful Environment Variables**

- `LLAVA_MODEL_SIZE`: `4B` or `8B`
- `LLAVA_MODEL_NAME`: explicit model id/path
- `LLAVA_LORA_RANKS`: ranks for training or inference sweep
- `LLAVA_CHECKPOINTS`: checkpoint numbers/names for inference
- `LLAVA_CHECKPOINT_ROOT`: adapter checkpoint root
- `LLAVA_TRAIN_JSONL`: training JSONL path
- `LLAVA_MAX_LENGTH`: training sequence length, default `8192`
- `LLAVA_MAX_VIDEOS`: max sampled videos per project
- `LLAVA_NUM_SAMPLED_FRAMES`: frames sampled from each video
- `LLAVA_MAX_NEW_TOKENS`: generation token limit
