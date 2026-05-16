**Qwen2-VL Workspace**

This folder is a lightweight companion to `internvl3_finetune` for running
`../inference_Qwen2-VL.py` and `../train_qwen2vl_sft.py`.

The training script uses the same JSONL records as the InternVL3 SFT scaffold:
`frame_paths` or `video_path`, plus `target_json`, `label` + `reason`, or
label-only targets.

**Dependencies**

LoRA training needs PEFT:

```bash
python -m pip install peft accelerate
```

`transformers` in this workspace is `4.45.2`, which has Qwen2-VL support.

**Training**

The launcher trains one LoRA adapter per combo. It looks for
`qwen2vl_finetune/train_full_${combo}.jsonl` first and falls back to the matching
files in `internvl3_finetune`.

```bash
cd /home/jamesyoung0623/Syn3D-Bench
qwen2vl_finetune/run_qwen2vl_sft.sh train
```

Useful overrides:

```bash
QWEN2VL_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct \
QWEN2VL_COMBOS=ST \
QWEN2VL_RUN_TAG=qwen2vl_2b_lora_16 \
QWEN2VL_LORA_R=16 \
QWEN2VL_EPOCHS=3 \
qwen2vl_finetune/run_qwen2vl_sft.sh train
```

Saved artifacts:
- `adapter/`
- `finetune_config.json`
- `finetune_manifest.json`
- `finetune_args.json`
- processor/tokenizer files

**Inference With A Finetuned Adapter**

```bash
cd /home/jamesyoung0623/Syn3D-Bench
QWEN2VL_RUN_TAG=qwen2vl_2b_lora_16 \
QWEN2VL_COMBOS=ST \
qwen2vl_finetune/run_qwen2vl_sft.sh infer
```

This mode also writes `train_exclude_object_ids.txt` from the selected training
JSONL files and skips those object IDs during inference.

Or explicitly:

```bash
QWEN2VL_MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct \
QWEN2VL_FINETUNE_DIR=qwen2vl_finetune/checkpoints/qwen2vl_2b_lora_16_ST_sft/checkpoint-final \
python inference_Qwen2-VL.py
```

**Quick Start**

```bash
cd /home/jamesyoung0623/Syn3D-Bench
qwen2vl_finetune/run_qwen2vl_inference.sh infer
```

`infer`, `inference`, and the no-argument default all skip objects found in
training JSONL files.

Default settings:
- model: `Qwen/Qwen2-VL-7B-Instruct`
- max videos per project: `100`
- sampled frames per video: `12`
- max output tokens: `128`
- output tag: `qwen2vl_7b`

For the 2B model:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
QWEN2VL_MODEL_SIZE=2B qwen2vl_finetune/run_qwen2vl_inference.sh infer
```

For a custom Hugging Face model id:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
QWEN2VL_MODEL_NAME=/path/to/or/hf/model \
QWEN2VL_MODEL_SIZE=7B \
QWEN2VL_RUN_TAG=my_qwen2vl_run \
qwen2vl_finetune/run_qwen2vl_inference.sh infer
```

**Exclude Training Objects**

Training-object exclusion is the default:

```bash
cd /home/jamesyoung0623/Syn3D-Bench
qwen2vl_finetune/run_qwen2vl_inference.sh
```

This writes `train_exclude_object_ids.txt` and sets
`QWEN2VL_EXCLUDE_OBJECT_IDS_PATH` before running inference. It looks for
`train_reason_replay_*.jsonl` and `train_full_*.jsonl` in this folder and
`../internvl3_finetune`.

To include all objects anyway, use the explicit opt-out:

```bash
qwen2vl_finetune/run_qwen2vl_inference.sh infer-all
```

You can also provide explicit JSONL files:

```bash
QWEN2VL_EXCLUDE_JSONLS="qwen2vl_finetune/my_train.jsonl,internvl3_finetune/train_full_ST.jsonl" \
qwen2vl_finetune/run_qwen2vl_inference.sh infer-excluding-train
```

**Useful Environment Variables**

- `QWEN2VL_MODEL_SIZE`: `2B` or `7B`
- `QWEN2VL_MODEL_NAME`: explicit model id/path
- `QWEN2VL_RUN_TAG`: suffix used in output filenames
- `QWEN2VL_MAX_VIDEOS`: max sampled videos per project
- `QWEN2VL_NUM_SAMPLED_FRAMES`: frames sampled from each video
- `QWEN2VL_MAX_NEW_TOKENS`: generation token limit
- `QWEN2VL_BATCH_SIZE`: inference batch size
- `QWEN2VL_DTYPE`: `auto`, `bf16`, or `fp16`
- `QWEN2VL_DEVICE_MAP`: default `auto`
- `QWEN2VL_LOAD_IN_8BIT`: `True` or `False`
- `QWEN2VL_OUTPUT_PROJECT_ROOT`: root containing `ULIP`, `InstantMesh`, `LGM`,
  `SAM3D`, `TRELLIS`, and `TRELLIS_text`

**Modes**

```bash
qwen2vl_finetune/run_qwen2vl_inference.sh infer
qwen2vl_finetune/run_qwen2vl_inference.sh prepare-exclude
qwen2vl_finetune/run_qwen2vl_inference.sh infer-excluding-train
qwen2vl_finetune/run_qwen2vl_inference.sh infer-all
```
