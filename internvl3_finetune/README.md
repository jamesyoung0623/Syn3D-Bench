**InternVL3 Finetuning**
This scaffold fine-tunes InternVL3 for the exact question format used by
`../inference_InternVL3.py`.

Recommended recipe:
- Freeze `vision_model`
- Fully train `mlp1` (the projector)
- Add LoRA on `language_model`

Files:
- `../train_internvl3_sft.py`
- `dataset_example.jsonl`

**Dependencies**
The default Python in this workspace currently has `transformers`, `torch`, and `datasets`, but not `peft`.

Install at least:
```bash
python -m pip install peft accelerate
```

If you want to reuse the finetuned LoRA adapter in inference, `peft` must also be installed in the inference environment.

**Dataset**
Use JSONL. Each line should contain either:
- `video_path`
- or `frame_paths`

And either:
- `target_json`
- or `label` only
- or `label` + `confidence` + `reasons`

Example:
```json
{"id":"instantmesh_02691156-10155655850468db78d106ce0a280f87","dataset":"InstantMesh","project":"InstantMesh","video_path":"/path/to/project/root/InstantMesh/outputs/videos_white/02691156-10155655850468db78d106ce0a280f87.mp4","label":"synthetic"}
```

If an example only has `label`, the scaffold trains on a label-only JSON target like:
```json
{"label":"synthetic"}
```

That is the right fit for your current annotations, but it mainly teaches the class decision. If you later add `confidence` and `reasons`, the same scaffold can train the full explanation format as well.

To build a larger label-only dataset over your project video roots:

```bash
cd /path/to/Syn3D-Bench
python -u internvl3_finetune/build_project_dataset.py \
  --output_jsonl internvl3_finetune/all_projects_label_only.jsonl \
  --shuffle
```

Useful options:
- `--max_per_project 10000` to cap each source during debugging
- `--val_fraction 0.1` to also write object-level train/val splits
- `--projects ShapeNet,InstantMesh,LGM` to restrict the source set

**Bootstrapping full JSON from label-only data**
If your current dataset only has `label`, you can generate pseudo-targets with a teacher InternVL3 model that sees both the images and the gold label:

```bash
cd /path/to/Syn3D-Bench
python -u internvl3_finetune/generate_label_conditioned_targets.py \
  --input_jsonl internvl3_finetune/all_projects_label_only.jsonl \
  --output_jsonl /abs/path/train_full.jsonl \
  --model_name OpenGVLab/InternVL3-8B \
  --image_size 448 \
  --num_sampled_frames 12
```

This writes one JSONL line per example and appends:
- `confidence`
- `reasons`
- `target_json`
- `teacher_model_name`
- `teacher_raw_output`

The label stays fixed to your provided gold label. The teacher only fills in support strength and reasons.

**Training**
Small-model debug run:
```bash
cd /path/to/Syn3D-Bench
python -u train_internvl3_sft.py \
  --model_name OpenGVLab/InternVL3-1B \
  --train_jsonl /abs/path/train.jsonl \
  --eval_jsonl /abs/path/val.jsonl \
  --output_dir /abs/path/checkpoints/internvl3_1b_sft \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --projector_learning_rate 2e-4 \
  --llm_learning_rate 2e-5 \
  --num_train_epochs 2 \
  --logging_steps 10 \
  --save_strategy epoch \
  --evaluation_strategy epoch \
  --bf16 True \
  --remove_unused_columns False
```

Recommended starting settings:
- `--model_name OpenGVLab/InternVL3-1B` or `2B` while validating the pipeline
- `--freeze_vision True`
- `--tune_projector True`
- `--llm_lora True`
- `--lora_r 16`
- `--lora_alpha 32`
- `--projector_learning_rate 2e-4`
- `--llm_learning_rate 2e-5`
- `--num_sampled_frames 12`
- `--image_size 448`
- `--max_tiles_per_frame 1`

**Saved artifacts**
The scaffold saves:
- `projector.pt`
- `llm_adapter/`
- `finetune_config.json`
- `finetune_manifest.json`
- `finetune_args.json`
- tokenizer files

**Inference with finetuned weights**
Point the current InternVL3 inference scripts to the saved artifacts:

```bash
INTERNVL3_MODEL_SIZE=1B \
INTERNVL3_FINETUNE_DIR=/abs/path/checkpoints/internvl3_1b_sft/checkpoint-final \
python inference_InternVL3.py
```

Or explicitly:
```bash
INTERNVL3_MODEL_NAME=OpenGVLab/InternVL3-1B \
INTERNVL3_PROJECTOR_PATH=/abs/path/checkpoints/internvl3_1b_sft/checkpoint-final/projector.pt \
INTERNVL3_LLM_ADAPTER_PATH=/abs/path/checkpoints/internvl3_1b_sft/checkpoint-final/llm_adapter \
python inference_InternVL3.py
```

**Notes**
- This scaffold uses the model’s real supervised `forward(...)` path, not `chat(...)`.
- It masks the prompt tokens and trains only on the assistant JSON answer.
- It uses separate optimizer groups for projector vs LLM LoRA so you can adapt the bridge faster than the decoder.
- If you save a LoRA adapter, the inference environment also needs `peft` installed.
