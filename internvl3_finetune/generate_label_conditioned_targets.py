import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, HfArgumentParser

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_common import setup_run_logging
from inference_InternVL3 import (
    build_frame_retry_schedule,
    build_greedy_generation_config,
    build_max_memory,
    build_split_device_map,
    build_transform,
    configure_model_for_image_size,
    dynamic_preprocess,
    is_cuda_oom_error,
    resolve_visual_device,
    sample_video_frames,
)


setup_run_logging("internvl3_pseudolabel", __file__)

VALID_LABELS = {"real", "synthetic", "uncertain"}


@dataclass
class PseudoLabelArguments:
    input_jsonl: str = ""
    output_jsonl: str = ""
    model_name: str = "OpenGVLab/InternVL3-8B"
    image_size: int = 448
    num_sampled_frames: int = 12
    minimum_num_sampled_frames: int = 1
    max_tiles_per_frame: int = 1
    use_thumbnail: bool = True
    max_new_tokens: int = 128
    device_map_mode: str = "auto"
    load_in_8bit: bool = False
    dtype: str = "auto"
    shuffle: bool = False
    sample_seed: int = 0
    max_examples: int = 0
    start_index: int = 0
    overwrite: bool = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        raise ValueError(f"No records found in {path}")
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def derive_record_id(example: dict[str, Any]) -> str:
    if "id" in example and str(example["id"]).strip():
        return str(example["id"])
    if "video_path" in example:
        return Path(example["video_path"]).stem
    if "frame_paths" in example and example["frame_paths"]:
        return Path(example["frame_paths"][0]).parent.name
    raise KeyError("Each record needs `id`, `video_path`, or `frame_paths`.")


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            target_json = record.get("target_json")
            has_full_target = isinstance(target_json, dict) and {"label", "reason"}.issubset(
                target_json.keys()
            )
            if has_full_target:
                completed_ids.add(derive_record_id(record))
    return completed_ids


def load_frames(example: dict[str, Any], num_sampled_frames: int) -> list[Image.Image]:
    if "frame_paths" in example:
        frames = []
        for frame_path in example["frame_paths"]:
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
        return frames
    if "video_path" in example:
        return sample_video_frames(Path(example["video_path"]), num_sampled_frames)
    raise KeyError("Each record must include either `video_path` or `frame_paths`.")


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


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(bf16_supported) and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError("--dtype must be one of: auto, bf16, fp16, fp32")


def load_model_and_tokenizer(args: PseudoLabelArguments):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    torch_dtype = torch.bfloat16 if args.load_in_8bit else resolve_torch_dtype(args.dtype)
    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_flash_attn": torch.cuda.is_available() and torch_dtype != torch.float32,
    }

    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    if torch.cuda.is_available():
        if args.device_map_mode == "split" and torch.cuda.device_count() > 1:
            device_map = build_split_device_map(args.model_name)
            load_kwargs["device_map"] = device_map
            print(f"Using split device_map across {torch.cuda.device_count()} GPUs.")
        else:
            max_memory = build_max_memory()
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
                print(f"Using max_memory={max_memory}")
            load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(args.model_name, **load_kwargs).eval()
    configure_model_for_image_size(model, args.image_size)
    return model, tokenizer, torch_dtype


def build_label_conditioned_prompt(label: str) -> str:
    return f"""
You are analyzing multiple rendered images of the same 3D asset.

The gold label for the asset's origin is fixed:
- "{label}"

Definitions:
- "real": primarily authored by a person through manual modeling, sculpting, CAD design, manual assembly, or substantial human cleanup.
- "synthetic": primarily produced by an automatic generative system such as text-to-3D, image-to-3D, reconstruction, diffusion-based generation, or procedural/generative modeling.
- "uncertain": the visible evidence is insufficient to reliably determine whether the asset was primarily human-authored or machine-generated.

Your task is NOT to predict the label. Your task is to explain why the visible evidence supports the provided label.

Important rules:
1. Keep the label exactly as provided.
2. Base your explanation only on diagnostic evidence about the underlying 3D asset's origin.
3. Ignore rendering consistency by itself. Multiple views, black backgrounds, or consistent lighting are not diagnostic on their own.
4. If the provided label is weakly supported by the visible evidence, explain the uncertainty or mismatch in the reason.
5. For "real", focus on signs of deliberate design intent, coherent part structure, functional details, and plausible geometry.
6. For "synthetic", focus on generative artifacts, repeated or nonsensical structure, over-smoothing, merged parts, or implausible geometry.
7. For "uncertain", explain why the visible evidence is not diagnostic.

Output requirements:
- Return valid JSON only.
- Use the provided label exactly.
- Provide exactly one reason in one sentence.
- The reason must cite a diagnostic visual cue.
- Do not mention that the label was given to you.

Return JSON with this schema:
{{
  "label": "{label}",
  "reason": "one-sentence reason"
}}
""".strip()


def build_question(num_frames: int, label: str) -> str:
    frame_placeholders = "\n".join(f"Frame {idx + 1}: <image>" for idx in range(num_frames))
    return f"{frame_placeholders}\n\n{build_label_conditioned_prompt(label)}"


def prepare_inputs(
    model,
    frames: list[Image.Image],
    image_size: int,
    max_tiles_per_frame: int,
    use_thumbnail: bool,
    torch_dtype: torch.dtype,
):
    transform = build_transform(image_size)
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
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    visual_device = resolve_visual_device(model)
    pixel_values = torch.cat(pixel_values_list, dim=0).to(
        device=visual_device,
        dtype=torch_dtype,
    )
    return pixel_values, num_patches_list


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_response(response: str) -> dict[str, Any]:
    candidates = []
    stripped = response.strip()
    if stripped:
        candidates.append(stripped)

    unfenced = strip_code_fences(response)
    if unfenced and unfenced not in candidates:
        candidates.append(unfenced)

    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        braced = response[start : end + 1].strip()
        if braced and braced not in candidates:
            candidates.append(braced)

    last_error = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                raise ValueError("Parsed JSON must be an object.")
            return parsed
        except Exception as error:
            last_error = error

    raise ValueError(f"Could not parse teacher response as JSON: {last_error}")


def normalize_reason(reason: Any) -> str:
    normalized = str(reason).strip()
    if normalized:
        return normalized
    raise ValueError("`reason` must be a non-empty string.")


def normalize_target_json(parsed: dict[str, Any], gold_label: str) -> tuple[dict[str, Any], str | None]:
    teacher_label = parsed.get("label")
    reason = normalize_reason(parsed["reason"])
    normalized = {
        "label": gold_label,
        "reason": reason,
    }
    teacher_label_mismatch = None
    if teacher_label is not None and str(teacher_label) != gold_label:
        teacher_label_mismatch = str(teacher_label)
    return normalized, teacher_label_mismatch


def annotate_example(model, tokenizer, example: dict[str, Any], args: PseudoLabelArguments, torch_dtype: torch.dtype) -> dict[str, Any]:
    gold_label = str(example["label"])
    if gold_label not in VALID_LABELS:
        raise ValueError(f"Unsupported label {gold_label!r}. Expected one of {sorted(VALID_LABELS)}.")

    all_frames = load_frames(example, args.num_sampled_frames)
    frame_retry_schedule = build_frame_retry_schedule(
        initial_num_frames=min(len(all_frames), args.num_sampled_frames),
        minimum_num_sampled_frames=min(args.minimum_num_sampled_frames, len(all_frames)),
    )
    generation_config = build_greedy_generation_config(max_new_tokens=args.max_new_tokens)

    for attempt_idx, current_num_frames in enumerate(frame_retry_schedule):
        pixel_values = None
        try:
            current_frames = select_frame_subset(all_frames, current_num_frames)
            pixel_values, num_patches_list = prepare_inputs(
                model=model,
                frames=current_frames,
                image_size=args.image_size,
                max_tiles_per_frame=args.max_tiles_per_frame,
                use_thumbnail=args.use_thumbnail,
                torch_dtype=torch_dtype,
            )
            question = build_question(len(current_frames), gold_label)

            with torch.inference_mode():
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    history=None,
                    return_history=False,
                    num_patches_list=num_patches_list,
                )

            parsed = parse_json_response(response)
            target_json, teacher_label_mismatch = normalize_target_json(parsed, gold_label)

            annotated = dict(example)
            annotated["label"] = gold_label
            annotated["reason"] = target_json["reason"]
            annotated["target_json"] = target_json
            annotated["teacher_model_name"] = args.model_name
            annotated["teacher_raw_output"] = response
            annotated["annotation_status"] = "ok"
            annotated["num_sampled_frames_used"] = len(current_frames)
            annotated["num_tiles"] = sum(num_patches_list)
            if teacher_label_mismatch is not None:
                annotated["teacher_label_mismatch"] = teacher_label_mismatch
            return annotated
        except Exception as error:
            is_last_attempt = attempt_idx == len(frame_retry_schedule) - 1
            if not is_cuda_oom_error(error) or is_last_attempt:
                failed = dict(example)
                failed["annotation_status"] = "error"
                failed["teacher_model_name"] = args.model_name
                failed["teacher_error"] = str(error)
                return failed

            next_num_frames = frame_retry_schedule[attempt_idx + 1]
            print(
                f"CUDA OOM for {derive_record_id(example)} with {current_num_frames} frames; "
                f"retrying with {next_num_frames} frames."
            )
        finally:
            del pixel_values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def resolve_output_path(args: PseudoLabelArguments) -> Path:
    input_path = Path(args.input_jsonl)
    if args.output_jsonl:
        return Path(args.output_jsonl)
    return input_path.with_name(f"{input_path.stem}_full.jsonl")


def main() -> None:
    parser = HfArgumentParser(PseudoLabelArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if not args.input_jsonl:
        raise ValueError("--input_jsonl is required.")
    if args.device_map_mode not in {"auto", "split"}:
        raise ValueError("--device_map_mode must be 'auto' or 'split'.")
    if args.minimum_num_sampled_frames < 1:
        raise ValueError("--minimum_num_sampled_frames must be >= 1.")
    if args.minimum_num_sampled_frames > args.num_sampled_frames:
        raise ValueError("--minimum_num_sampled_frames cannot exceed --num_sampled_frames.")

    input_path = Path(args.input_jsonl)
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    examples = load_jsonl(input_path)
    completed_ids = load_completed_ids(output_path)
    pending_examples = [example for example in examples if derive_record_id(example) not in completed_ids]

    if args.shuffle:
        pending_examples = pending_examples[:]
        random.Random(args.sample_seed).shuffle(pending_examples)

    if args.start_index:
        pending_examples = pending_examples[args.start_index :]
    if args.max_examples > 0:
        pending_examples = pending_examples[: args.max_examples]

    print("Pseudo-label generation settings:")
    print(f"  input_jsonl={input_path}")
    print(f"  output_jsonl={output_path}")
    print(f"  model_name={args.model_name}")
    print(f"  image_size={args.image_size}")
    print(f"  num_sampled_frames={args.num_sampled_frames}")
    print(f"  max_tiles_per_frame={args.max_tiles_per_frame}")
    print(f"  pending_examples={len(pending_examples)}")

    if not pending_examples:
        print("No pending examples to annotate.")
        return

    model, tokenizer, torch_dtype = load_model_and_tokenizer(args)

    for example_idx, example in enumerate(pending_examples, start=1):
        record_id = derive_record_id(example)
        print(f"\n[{example_idx}/{len(pending_examples)}] Annotating {record_id}")
        annotated = annotate_example(
            model=model,
            tokenizer=tokenizer,
            example=example,
            args=args,
            torch_dtype=torch_dtype,
        )
        append_jsonl(output_path, annotated)
        print(f"  status={annotated['annotation_status']}")
        if annotated["annotation_status"] == "ok":
            print(f"  target_json={annotated['target_json']}")
        else:
            print(f"  error={annotated['teacher_error']}")

    print(f"\nDone. Saved pseudo-annotations to {output_path}")


if __name__ == "__main__":
    main()
