import json
import os
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import torch
import torch.utils.checkpoint as torch_checkpoint
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_common import setup_run_logging
from inference_InternVL3 import (
    build_prompt,
    build_transform,
    configure_model_for_image_size,
    dynamic_preprocess,
    sample_video_frames,
)


setup_run_logging("internvl3_train", __file__)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def patch_checkpoint_use_reentrant_default() -> None:
    if getattr(torch_checkpoint.checkpoint, "_internvl3_use_reentrant_patched", False):
        return

    original_checkpoint = torch_checkpoint.checkpoint

    def checkpoint_with_explicit_use_reentrant(function, *args, **kwargs):
        kwargs.setdefault("use_reentrant", False)
        return original_checkpoint(function, *args, **kwargs)

    checkpoint_with_explicit_use_reentrant._internvl3_use_reentrant_patched = True
    torch_checkpoint.checkpoint = checkpoint_with_explicit_use_reentrant


def configure_vision_gradient_checkpointing(model, enabled: bool) -> None:
    vision_encoder = getattr(getattr(model, "vision_model", None), "encoder", None)
    if hasattr(vision_encoder, "gradient_checkpointing"):
        vision_encoder.gradient_checkpointing = enabled


@dataclass
class ModelDataArguments:
    model_name: str = "OpenGVLab/InternVL3_5-2B-Instruct"
    train_jsonl: str = "internvl3_finetune/train_full_ST_consensus_uncertain.jsonl"
    eval_jsonl: str | None = None
    init_finetune_dir: str = ""
    init_projector_path: str = ""
    init_llm_adapter_path: str = ""
    image_size: int = 448
    num_sampled_frames: int = 12
    max_tiles_per_frame: int = 1
    use_thumbnail: bool = True
    max_length: int = 4096
    freeze_vision: bool = True
    tune_projector: bool = True
    llm_lora: bool = True
    lora_r: int = 4
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    projector_learning_rate: float = 5e-5
    llm_learning_rate: float = 1e-5


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
    if label not in {"real", "synthetic", "uncertain"}:
        raise ValueError(f"Unsupported label: {label!r}")

    if "reason" in target_json:
        reason = str(target_json["reason"]).strip()
        if not reason:
            raise ValueError("`reason` must be a non-empty string.")
        return json.dumps({"label": label, "reason": reason}, ensure_ascii=False)

    if "confidence" not in target_json and "reasons" not in target_json:
        return json.dumps({"label": label}, ensure_ascii=False)
    if "confidence" not in target_json or "reasons" not in target_json:
        raise ValueError("Provide both `confidence` and `reasons`, or provide neither for label-only training.")

    confidence = float(target_json["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be between 0 and 1, got {confidence}.")

    reasons = target_json["reasons"]
    if not isinstance(reasons, list) or len(reasons) != 3:
        raise ValueError("`reasons` must be a list of exactly 3 strings.")

    payload = {
        "label": label,
        "confidence": confidence,
        "reasons": [str(reason) for reason in reasons],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_question(num_frames: int) -> str:
    frame_placeholders = "\n".join(f"Frame {idx + 1}: <image>" for idx in range(num_frames))
    return f"{frame_placeholders}\n\n{build_prompt()}"


def expand_image_placeholders(text: str, num_patches_list: list[int], num_image_token: int) -> str:
    expanded = text
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
        expanded = expanded.replace("<image>", image_tokens, 1)
    return expanded


def build_prompt_and_response_texts(model, question: str, answer: str, num_patches_list: list[int]) -> tuple[str, str]:
    prompt_template = model.conv_template.copy()
    prompt_template.system_message = model.system_message
    prompt_template.append_message(prompt_template.roles[0], question)
    prompt_template.append_message(prompt_template.roles[1], None)
    prompt_text = prompt_template.get_prompt()

    response_template = model.conv_template.copy()
    response_template.system_message = model.system_message
    response_template.append_message(response_template.roles[0], question)
    response_template.append_message(response_template.roles[1], answer)
    full_text = response_template.get_prompt()

    prompt_text = expand_image_placeholders(prompt_text, num_patches_list, model.num_image_token)
    full_text = expand_image_placeholders(full_text, num_patches_list, model.num_image_token)
    return prompt_text, full_text


def longest_common_prefix(prompt_ids: list[int], full_ids: list[int]) -> int:
    prefix_length = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        prefix_length += 1
    return prefix_length


def load_frames(example: dict[str, Any], num_sampled_frames: int) -> list[Image.Image]:
    if "frame_paths" in example:
        frames = []
        for frame_path in example["frame_paths"]:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
        return frames
    if "video_path" in example:
        return sample_video_frames(Path(example["video_path"]), num_sampled_frames)
    raise KeyError("Each training example must include either `video_path` or `frame_paths`.")


def preprocess_frames(
    frames: list[Image.Image],
    transform,
    image_size: int,
    max_tiles_per_frame: int,
    use_thumbnail: bool,
) -> tuple[torch.Tensor, list[int]]:
    pixel_values_list = []
    num_patches_list = []
    for frame in frames:
        tiles = dynamic_preprocess(
            image=frame,
            min_num=1,
            max_num=max_tiles_per_frame,
            image_size=image_size,
            use_thumbnail=use_thumbnail,
        )
        tiled_tensor = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(tiled_tensor)
        num_patches_list.append(tiled_tensor.shape[0])

    return torch.cat(pixel_values_list, dim=0), num_patches_list


class InternVL3SFTDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        model,
        args: ModelDataArguments,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.transform = build_transform(args.image_size)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        frames = load_frames(example, self.args.num_sampled_frames)
        pixel_values, num_patches_list = preprocess_frames(
            frames=frames,
            transform=self.transform,
            image_size=self.args.image_size,
            max_tiles_per_frame=self.args.max_tiles_per_frame,
            use_thumbnail=self.args.use_thumbnail,
        )

        question = build_question(len(frames))
        answer = format_target_json(example)
        prompt_text, full_text = build_prompt_and_response_texts(
            model=self.model,
            question=question,
            answer=answer,
            num_patches_list=num_patches_list,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.args.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.args.max_length,
        )["input_ids"]

        prefix_length = longest_common_prefix(prompt_ids, full_ids)
        labels = [-100] * prefix_length + full_ids[prefix_length:]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_flags": torch.ones(pixel_values.shape[0], 1, dtype=torch.long),
        }


class InternVL3DataCollator:
    def __init__(self, tokenizer, pixel_dtype: torch.dtype):
        self.tokenizer = tokenizer
        self.pixel_dtype = pixel_dtype

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before collation.")

        input_ids = []
        labels = []
        attention_mask = []
        pixel_values = []
        image_flags = []

        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_length - seq_len
            input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            attention_mask.append([1] * seq_len + [0] * pad_len)
            pixel_values.append(feature["pixel_values"])
            image_flags.append(feature["image_flags"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "pixel_values": torch.cat(pixel_values, dim=0).to(dtype=self.pixel_dtype),
            "image_flags": torch.cat(image_flags, dim=0),
        }


def freeze_model_for_projector_plus_llm_lora(model, args: ModelDataArguments) -> None:
    model.requires_grad_(False)

    if not args.freeze_vision:
        model.vision_model.requires_grad_(True)

    if args.tune_projector:
        model.mlp1.requires_grad_(True)


def resolve_initial_finetune_paths(args: ModelDataArguments) -> tuple[str, str]:
    projector_path = args.init_projector_path.strip()
    llm_adapter_path = args.init_llm_adapter_path.strip()

    if args.init_finetune_dir.strip():
        init_dir = Path(args.init_finetune_dir)
        if not projector_path:
            candidate = init_dir / "projector.pt"
            if candidate.exists():
                projector_path = str(candidate)
        if not llm_adapter_path:
            candidate = init_dir / "llm_adapter"
            if candidate.exists():
                llm_adapter_path = str(candidate)

    return projector_path, llm_adapter_path


def load_initial_projector_weights(model, args: ModelDataArguments) -> None:
    projector_path, _ = resolve_initial_finetune_paths(args)
    if not projector_path:
        return

    projector_state = torch.load(projector_path, map_location="cpu")
    model.mlp1.load_state_dict(projector_state, strict=True)
    print(f"Loaded initial finetuned projector from {projector_path}")


def attach_llm_lora(model, args: ModelDataArguments):
    if not args.llm_lora:
        return model

    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LLM LoRA requires `peft`. Install it with `python -m pip install peft`."
        ) from exc

    _, llm_adapter_path = resolve_initial_finetune_paths(args)
    if llm_adapter_path:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            llm_adapter_path,
            is_trainable=True,
        )
        model.language_model.print_trainable_parameters()
        print(f"Loaded initial trainable LLM adapter from {llm_adapter_path}")
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
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()
    return model


def maybe_force_single_gpu_trainer(training_args: TrainingArguments) -> None:
    if not torch.cuda.is_available():
        return

    local_rank = getattr(training_args, "local_rank", -1)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    allow_dataparallel = os.environ.get("INTERNVL3_TRAIN_ALLOW_DATAPARALLEL", "").strip().lower() in {
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


def save_finetuned_artifacts(model, tokenizer, output_dir: str, model_name: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)

    torch.save(model.mlp1.state_dict(), output_path / "projector.pt")

    llm_adapter_dir = output_path / "llm_adapter"
    if hasattr(model.language_model, "peft_config") and hasattr(model.language_model, "save_pretrained"):
        try:
            model.language_model.save_pretrained(llm_adapter_dir, save_embedding_layers=False)
            saved_llm_adapter = True
        except Exception:
            saved_llm_adapter = False
    else:
        saved_llm_adapter = False

    finetune_config = {
        "base_model": model_name,
        "projector_path": "projector.pt",
        "llm_adapter_path": "llm_adapter" if saved_llm_adapter else None,
        "note": "Load with INTERNVL3_FINETUNE_DIR or the explicit adapter/projector env vars.",
    }
    with open(output_path / "finetune_config.json", "w", encoding="utf-8") as handle:
        json.dump(finetune_config, handle, indent=2, ensure_ascii=False)
    with open(output_path / "finetune_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(finetune_config, handle, indent=2, ensure_ascii=False)


class InternVL3SFTTrainer(Trainer):
    def __init__(self, *args, model_args: ModelDataArguments, artifact_tokenizer=None, **kwargs) -> None:
        self.artifact_tokenizer = artifact_tokenizer
        super().__init__(*args, **kwargs)
        self.model_args = model_args

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        projector_params = []
        other_trainable_params = []
        projector_param_ids = {
            id(param) for param in self.model.mlp1.parameters() if param.requires_grad
        }

        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if id(param) in projector_param_ids:
                projector_params.append(param)
            else:
                other_trainable_params.append(param)

        optimizer_grouped_parameters = []
        if projector_params:
            optimizer_grouped_parameters.append(
                {
                    "params": projector_params,
                    "lr": self.model_args.projector_learning_rate,
                    "weight_decay": self.args.weight_decay,
                }
            )
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
            tokenizer=self.artifact_tokenizer,
            output_dir=output_dir,
            model_name=self.model.config._name_or_path,
        )
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        with open(os.path.join(output_dir, "finetune_args.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(self.model_args), handle, indent=2, ensure_ascii=False)


def main() -> None:
    patch_checkpoint_use_reentrant_default()

    parser = HfArgumentParser((ModelDataArguments, ScriptTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    if not model_args.train_jsonl:
        raise ValueError("--train_jsonl is required.")
    if not model_args.tune_projector and not model_args.llm_lora and model_args.freeze_vision:
        raise ValueError("Nothing is trainable. Enable projector tuning, LLM LoRA, or unfreeze vision.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if training_args.bf16:
        training_dtype = torch.bfloat16
    elif training_args.fp16:
        training_dtype = torch.float16
    else:
        training_dtype = torch.float32

    model = AutoModel.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        torch_dtype=training_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=training_dtype != torch.float32,
    )
    configure_model_for_image_size(model, model_args.image_size)
    configure_vision_gradient_checkpointing(
        model,
        enabled=bool(training_args.gradient_checkpointing and not model_args.freeze_vision),
    )
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.config.use_cache = False

    load_initial_projector_weights(model, model_args)
    freeze_model_for_projector_plus_llm_lora(model, model_args)
    attach_llm_lora(model, model_args)

    if training_args.gradient_checkpointing:
        if hasattr(model.language_model, "gradient_checkpointing_enable"):
            try:
                model.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.language_model.gradient_checkpointing_enable()
        elif hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()

        if hasattr(model.language_model, "enable_input_require_grads"):
            model.language_model.enable_input_require_grads()
        elif hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return_only_loss_during_training(model)
    print_trainable_parameter_summary(model)

    train_examples = load_jsonl(model_args.train_jsonl)
    eval_examples = load_jsonl(model_args.eval_jsonl) if model_args.eval_jsonl else None

    train_dataset = InternVL3SFTDataset(train_examples, tokenizer, model, model_args)
    eval_dataset = (
        InternVL3SFTDataset(eval_examples, tokenizer, model, model_args)
        if eval_examples is not None
        else None
    )

    data_collator = InternVL3DataCollator(tokenizer, pixel_dtype=training_dtype)
    maybe_force_single_gpu_trainer(training_args)

    trainer_class_params = inspect.signature(Trainer.__init__).parameters
    trainer_tokenizer_kwargs = (
        {"processing_class": tokenizer}
        if "processing_class" in trainer_class_params
        else {"tokenizer": tokenizer}
    )

    trainer = InternVL3SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        model_args=model_args,
        artifact_tokenizer=tokenizer,
        **trainer_tokenizer_kwargs,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if trainer.is_world_process_zero():
        final_output_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        save_finetuned_artifacts(model, tokenizer, final_output_dir, model_args.model_name)
        torch.save(training_args, os.path.join(final_output_dir, "training_args.bin"))
        with open(os.path.join(final_output_dir, "finetune_args.json"), "w", encoding="utf-8") as handle:
            json.dump(asdict(model_args), handle, indent=2, ensure_ascii=False)
        print(f"Saved final finetuned artifacts to {final_output_dir}")


if __name__ == "__main__":
    main()
