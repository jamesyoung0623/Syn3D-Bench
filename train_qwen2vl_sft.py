import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
import torch.utils.checkpoint as torch_checkpoint
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT_DIR = Path(__file__).resolve().parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_common import setup_run_logging


setup_run_logging("qwen2vl_train", __file__)

VALID_LABELS = {"real", "synthetic", "uncertain"}
IMAGE_PAD_TOKEN = "<|image_pad|>"
PLACEHOLDER_TOKEN = "<|placeholder|>"


@dataclass
class ModelDataArguments:
    model_name: str = "OpenGVLab/InternVL3_5-2B-Instruct"
    train_jsonl: str = "internvl3_finetune/train_full_ST_consensus_uncertain.jsonl"
    eval_jsonl: str | None = None
    init_finetune_dir: str = ""
    init_adapter_path: str = ""
    frame_max_pixels: int = 256 * 28 * 28
    frame_min_pixels: int = 4 * 28 * 28
    num_sampled_frames: int = 12
    max_length: int = 4096
    tune_language_model: bool = False
    llm_lora: bool = True
    lora_r: int = 4
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    llm_learning_rate: float = 1e-5
    attn_implementation: str = ""


@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = "internvl3_finetune/checkpoints/internvl3_2b_lora_4_ST_consensus_uncertain_reason_sft"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 10
    logging_steps: float = 1
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.1


def patch_checkpoint_use_reentrant_default() -> None:
    if getattr(torch_checkpoint.checkpoint, "_qwen2vl_use_reentrant_patched", False):
        return

    original_checkpoint = torch_checkpoint.checkpoint

    def checkpoint_with_explicit_use_reentrant(function, *args, **kwargs):
        kwargs["use_reentrant"] = False
        return original_checkpoint(function, *args, **kwargs)

    checkpoint_with_explicit_use_reentrant._qwen2vl_use_reentrant_patched = True
    torch_checkpoint.checkpoint = checkpoint_with_explicit_use_reentrant


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_idx} of {path}: {exc}") from exc
    if not records:
        raise ValueError(f"No training records found in {path}")
    return records


def format_target_json(example: dict[str, Any]) -> str:
    if "target_json" in example:
        target_json = example["target_json"]
    elif "label" in example and "reason" in example:
        target_json = {
            "label": example["label"],
            "reason": example["reason"],
        }
    elif "label" in example and "confidence" not in example and "reasons" not in example:
        target_json = {
            "label": example["label"],
        }
    else:
        for key in ("label", "confidence", "reasons"):
            if key not in example:
                raise KeyError(f"Training example must contain `target_json` or `{key}`.")
        target_json = {
            "label": example["label"],
            "confidence": float(example["confidence"]),
            "reasons": example["reasons"],
        }

    label = str(target_json["label"])
    if label not in VALID_LABELS:
        raise ValueError(f"Unsupported label: {label!r}")

    if "reason" in target_json:
        reason = str(target_json["reason"]).strip()
        if not reason:
            raise ValueError("`reason` must be a non-empty string.")
        return json.dumps({"label": label, "reason": reason}, ensure_ascii=False)

    if "confidence" not in target_json and "reasons" not in target_json:
        return json.dumps({"label": label}, ensure_ascii=False)
    if "confidence" not in target_json or "reasons" not in target_json:
        raise ValueError("Provide both `confidence` and `reasons`, or neither for label-only training.")

    confidence = float(target_json["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}.")

    reasons = target_json["reasons"]
    if not isinstance(reasons, list) or len(reasons) != 3:
        raise ValueError("`reasons` must be a list of exactly 3 strings.")

    return json.dumps(
        {
            "label": label,
            "confidence": confidence,
            "reasons": [str(reason) for reason in reasons],
        },
        ensure_ascii=False,
    )


def build_prompt() -> str:
    return """
You are analyzing multiple rendered images of the same 3D asset.

Your goal is to infer the origin of the underlying 3D model, not to describe the rendered views.

Decide whether the underlying 3D model is:
- "real"
- "synthetic"
- "uncertain"

Definitions:

- "real":
  The underlying 3D asset was primarily authored by a person through manual modeling, sculpting, CAD design, manual assembly, or substantial human editing/cleanup. A human determined most of the geometry, part structure, and important design details.

- "synthetic":
  The underlying 3D asset was primarily produced by an automatic generative system such as text-to-3D, image-to-3D, reconstruction, diffusion-based generation, or procedural/generative modeling, with little or no substantial manual correction. The machine determined most of the geometry and structure.

- "uncertain":
  The visible evidence is insufficient to reliably determine whether the asset was primarily human-authored or machine-generated.

Important rules:
1. Base your decision only on diagnostic evidence about the 3D model's origin.
2. Do NOT use the same kind of reason to justify opposite labels.
3. Focus on features of the underlying asset such as:
   - topology plausibility
   - part coherence
   - symmetry vs over-regularization
   - repeated geometry
   - implausible structure
   - missing functional details
   - texture/material inconsistency
   - semantic incoherence
   - signs of manual design intent
   - signs of procedural/generative artifacts
4. Ignore the fact that these are rendered images by themselves. Multiple views, consistent camera, or consistent lighting are NOT sufficient evidence for either class.
5. For "real", the reason should point to evidence of deliberate manual design, functional structure, meaningful detail placement, or coherent asset construction.
6. For "synthetic", the reason should point to evidence of generative artifacts, implausible geometry, repeated or nonsensical structure, over-smoothing, inconsistent semantics, or missing/merged functional parts.
7. For "uncertain", the reason should explain exactly why the visible evidence is not diagnostic.

Output requirements:
- Return valid JSON only.
- Provide exactly one reason in one sentence.
- The reason must cite a diagnostic visual cue.
- Do not repeat the label wording in the reason.

Return JSON with this schema:
{
  "label": "real" | "synthetic" | "uncertain",
  "reason": "one-sentence reason"
}
""".strip()


def sample_video_frames(video_path: Path, num_frames: int) -> list[Image.Image]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            raise ValueError(
                f"Video only has {total_frames} frames, cannot sample {num_frames} unique frames."
            )

        frame_indices = np.linspace(0, total_frames - 1, num_frames).round().astype(int)
        frame_indices = np.unique(frame_indices)
        if len(frame_indices) != num_frames:
            raise ValueError(
                f"Could not sample {num_frames} unique frames from a {total_frames}-frame video."
            )

        frames = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame_bgr = capture.read()
            if not success:
                raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        return frames
    finally:
        capture.release()


def select_frame_subset(frames: list[Image.Image], target_count: int) -> list[Image.Image]:
    if len(frames) <= target_count:
        return frames

    frame_indices = np.linspace(0, len(frames) - 1, target_count).round().astype(int)
    frame_indices = np.unique(frame_indices)
    if len(frame_indices) != target_count:
        raise ValueError(
            f"Could not downsample {len(frames)} frames to {target_count} unique frames."
        )
    return [frames[int(index)] for index in frame_indices]


def load_frames(example: dict[str, Any], num_sampled_frames: int) -> list[Image.Image]:
    if "frame_paths" in example:
        frames = []
        for frame_path in example["frame_paths"]:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
        return select_frame_subset(frames, num_sampled_frames)
    if "video_path" in example:
        return sample_video_frames(Path(example["video_path"]), num_sampled_frames)
    raise KeyError("Each training example must include either `video_path` or `frame_paths`.")


def build_user_message(num_images: int) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            [{"type": "image"} for _ in range(num_images)]
            + [{"type": "text", "text": build_prompt()}]
        ),
    }


def build_prompt_text(processor, num_images: int) -> str:
    return processor.apply_chat_template(
        [build_user_message(num_images)],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_text(processor, num_images: int, answer: str) -> str:
    return processor.apply_chat_template(
        [
            build_user_message(num_images),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def expand_image_pad_tokens(text: str, image_grid_thw: torch.Tensor, merge_size: int) -> str:
    expanded = text
    merge_length = merge_size**2
    for grid in image_grid_thw:
        repeat_count = int(grid.prod().item() // merge_length)
        expanded = expanded.replace(IMAGE_PAD_TOKEN, PLACEHOLDER_TOKEN * repeat_count, 1)
    return expanded.replace(PLACEHOLDER_TOKEN, IMAGE_PAD_TOKEN)


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    prefix_length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        prefix_length += 1
    return prefix_length


class Qwen2VLSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], processor, args: ModelDataArguments) -> None:
        self.examples = examples
        self.processor = processor
        self.args = args

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        frames = load_frames(example, self.args.num_sampled_frames)
        answer = format_target_json(example)

        return {
            "frames": frames,
            "prompt_text": build_prompt_text(self.processor, len(frames)),
            "full_text": build_full_text(self.processor, len(frames), answer),
        }


class Qwen2VLDataCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        full_texts = [feature["full_text"] for feature in features]
        prompt_texts = [feature["prompt_text"] for feature in features]
        frames_batch = [feature["frames"] for feature in features]
        image_counts = [len(frames) for frames in frames_batch]

        inputs = self.processor(
            text=full_texts,
            images=frames_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("Qwen2-VL processor did not return `image_grid_thw`.")

        merge_size = int(getattr(self.processor.image_processor, "merge_size", 2))
        expanded_prompt_texts = []
        grid_offset = 0
        for prompt_text, image_count in zip(prompt_texts, image_counts):
            grid_slice = image_grid_thw[grid_offset:grid_offset + image_count]
            expanded_prompt_texts.append(expand_image_pad_tokens(prompt_text, grid_slice, merge_size))
            grid_offset += image_count

        prompt_inputs = self.processor.tokenizer(
            expanded_prompt_texts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = input_ids.clone()

        for item_idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
            seq_len = int(attention_mask[item_idx].sum().item())
            full_ids = input_ids[item_idx, :seq_len].tolist()
            prefix_length = longest_common_prefix(prompt_ids, full_ids)

            if prefix_length >= seq_len:
                raise ValueError(
                    "The assistant target was fully masked or truncated. "
                    "Increase --max_length or reduce --num_sampled_frames/frame_max_pixels."
                )

            labels[item_idx, :prefix_length] = -100
            labels[item_idx, seq_len:] = -100

        inputs["labels"] = labels
        return inputs


def resolve_initial_adapter_path(args: ModelDataArguments) -> str:
    adapter_path = args.init_adapter_path.strip()
    if adapter_path:
        return adapter_path

    if not args.init_finetune_dir.strip():
        return ""

    init_dir = Path(args.init_finetune_dir)
    for candidate in (init_dir / "adapter", init_dir):
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    return ""


def load_model(args: ModelDataArguments, torch_dtype: torch.dtype):
    load_errors = []
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation.strip():
        load_kwargs["attn_implementation"] = args.attn_implementation.strip()

    for model_cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            return model_cls.from_pretrained(args.model_name, **load_kwargs)
        except Exception as exc:
            load_errors.append(f"{model_cls.__name__}: {exc}")

    raise RuntimeError(
        "Failed to load Qwen2-VL for training with the installed transformers setup.\n"
        + "\n".join(load_errors)
    )


def freeze_base_parameters(model, args: ModelDataArguments) -> None:
    model.requires_grad_(False)

    if args.tune_language_model:
        if hasattr(model, "model"):
            model.model.requires_grad_(True)
        if hasattr(model, "lm_head"):
            model.lm_head.requires_grad_(True)


def attach_llm_lora(model, args: ModelDataArguments):
    if not args.llm_lora:
        return model

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "Qwen2-VL LoRA fine-tuning requires `peft`. Install it with "
            "`python -m pip install peft accelerate`."
        ) from exc

    adapter_path = resolve_initial_adapter_path(args)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        model.print_trainable_parameters()
        print(f"Loaded initial trainable Qwen2-VL adapter from {adapter_path}")
        return model

    target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def maybe_enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def maybe_force_single_gpu_trainer(training_args: TrainingArguments) -> None:
    if not torch.cuda.is_available():
        return

    local_rank = getattr(training_args, "local_rank", -1)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    allow_dataparallel = os.environ.get("QWEN2VL_TRAIN_ALLOW_DATAPARALLEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if torch.cuda.device_count() > 1 and local_rank == -1 and world_size == 1 and not allow_dataparallel:
        training_args._n_gpu = 1
        print(
            "Multiple GPUs detected; forcing single-GPU Trainer mode to avoid DataParallel instability. "
            "Use torchrun/accelerate launch for real multi-GPU training."
        )


def print_trainable_parameter_summary(model) -> None:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
    pct = 100.0 * trainable_params / max(total_params, 1)
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({pct:.4f}%)"
    )


def return_only_loss_during_training(model) -> None:
    original_forward = model.forward

    def forward_without_training_logits(*args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = original_forward(*args, **kwargs)
        if model.training and kwargs.get("labels") is not None:
            if isinstance(outputs, dict):
                return {"loss": outputs["loss"]}
            return {"loss": outputs.loss}
        return outputs

    model.forward = forward_without_training_logits


def model_has_peft_adapter(model) -> bool:
    return hasattr(model, "peft_config") or hasattr(getattr(model, "base_model", None), "peft_config")


def save_finetuned_artifacts(model, processor, output_dir: str, model_args: ModelDataArguments) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_path)

    saved_adapter_path = None
    if model_has_peft_adapter(model) and hasattr(model, "save_pretrained"):
        adapter_dir = output_path / "adapter"
        model.save_pretrained(adapter_dir, save_embedding_layers=False)
        saved_adapter_path = "adapter"
    else:
        model.save_pretrained(output_path, safe_serialization=True)

    finetune_config = {
        "base_model": model_args.model_name,
        "adapter_path": saved_adapter_path,
        "frame_max_pixels": model_args.frame_max_pixels,
        "frame_min_pixels": model_args.frame_min_pixels,
        "num_sampled_frames": model_args.num_sampled_frames,
        "note": "Load with QWEN2VL_FINETUNE_DIR or QWEN2VL_ADAPTER_PATH.",
    }
    with open(output_path / "finetune_config.json", "w", encoding="utf-8") as handle:
        json.dump(finetune_config, handle, indent=2, ensure_ascii=False)
    with open(output_path / "finetune_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(finetune_config, handle, indent=2, ensure_ascii=False)


class Qwen2VLSFTTrainer(Trainer):
    def __init__(self, *args, model_args: ModelDataArguments, artifact_processor=None, **kwargs) -> None:
        self.artifact_processor = artifact_processor
        super().__init__(*args, **kwargs)
        self.model_args = model_args

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        other_trainable_params = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            other_trainable_params.append(param)

        optimizer_grouped_parameters = []
        if other_trainable_params:
            optimizer_grouped_parameters.append(
                {
                    "params": other_trainable_params,
                    "lr": self.model_args.llm_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )

        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters were enabled for fine-tuning.")

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_finetuned_artifacts(
            model=self.model,
            processor=self.artifact_processor,
            output_dir=output_dir,
            model_args=self.model_args,
        )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "finetune_args.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(self.model_args), handle, indent=2, ensure_ascii=False)


def main() -> None:
    patch_checkpoint_use_reentrant_default()

    parser = HfArgumentParser((ModelDataArguments, ScriptTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    set_seed(training_args.seed)

    if not model_args.train_jsonl:
        raise ValueError("--train_jsonl is required.")
    if model_args.frame_max_pixels < 1:
        raise ValueError("--frame_max_pixels must be >= 1.")
    if model_args.frame_min_pixels < 1:
        raise ValueError("--frame_min_pixels must be >= 1.")
    if model_args.num_sampled_frames < 1:
        raise ValueError("--num_sampled_frames must be >= 1.")
    if model_args.max_length < 1:
        raise ValueError("--max_length must be >= 1.")
    if not model_args.tune_language_model and not model_args.llm_lora:
        raise ValueError("Nothing is trainable. Enable --llm_lora or --tune_language_model.")

    if training_args.bf16:
        training_dtype = torch.bfloat16
    elif training_args.fp16:
        training_dtype = torch.float16
    else:
        training_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        min_pixels=model_args.frame_min_pixels,
        max_pixels=model_args.frame_max_pixels,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = load_model(model_args, training_dtype)
    if hasattr(model, "config"):
        model.config.use_cache = False

    freeze_base_parameters(model, model_args)
    model = attach_llm_lora(model, model_args)

    if training_args.gradient_checkpointing:
        maybe_enable_gradient_checkpointing(model)

    return_only_loss_during_training(model)
    print_trainable_parameter_summary(model)

    train_examples = load_jsonl(model_args.train_jsonl)
    eval_examples = load_jsonl(model_args.eval_jsonl) if model_args.eval_jsonl else None

    train_dataset = Qwen2VLSFTDataset(train_examples, processor, model_args)
    eval_dataset = (
        Qwen2VLSFTDataset(eval_examples, processor, model_args)
        if eval_examples is not None
        else None
    )
    data_collator = Qwen2VLDataCollator(processor, max_length=model_args.max_length)
    maybe_force_single_gpu_trainer(training_args)

    trainer_class_params = inspect.signature(Trainer.__init__).parameters
    trainer_processor_kwargs = (
        {"processing_class": processor}
        if "processing_class" in trainer_class_params
        else {"tokenizer": processor.tokenizer}
    )

    trainer = Qwen2VLSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_args=model_args,
        artifact_processor=processor,
        **trainer_processor_kwargs,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        final_output_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        save_finetuned_artifacts(model, processor, final_output_dir, model_args)
        torch.save(training_args, os.path.join(final_output_dir, "training_args.bin"))
        with open(os.path.join(final_output_dir, "finetune_args.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(model_args), handle, indent=2, ensure_ascii=False)
        print(f"Saved final finetuned artifacts to {final_output_dir}")


if __name__ == "__main__":
    main()
