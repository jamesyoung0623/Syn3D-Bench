import os
import gc
import json
import logging
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor

from inference_common import PROJECT_NAMES, project_root, project_videos_root, setup_run_logging


setup_run_logging("qwen2vl", __file__)


SUPPORTED_QWEN2VL_MODELS = {
    "2B": "Qwen/Qwen2-VL-2B-Instruct",
    "7B": "Qwen/Qwen2-VL-7B-Instruct",
}

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class ProjectDefaults:
    videos_root: Path
    output_dir: Path
    default_model_size: str = "7B"
    num_sampled_frames: int = 12
    frame_max_pixels: int = 256 * 28 * 28
    batch_size: int = 1
    max_videos: int = 1000
    sample_seed: int = 0
    minimum_num_sampled_frames: int = 1
    max_new_tokens: int = 128
    device_map: str = "auto"
    load_in_8bit: bool = False


@dataclass(frozen=True)
class RuntimeSettings:
    model_name: str
    model_size: str
    adapter_path: Path | None
    videos_root: Path
    output_path: Path
    exclude_object_ids_path: Path | None
    num_sampled_frames: int
    frame_max_pixels: int
    batch_size: int
    max_videos: int
    sample_seed: int
    minimum_num_sampled_frames: int
    max_new_tokens: int
    device_map: str
    load_in_8bit: bool
    torch_dtype: torch.dtype | str


class IgnoreAccelerateP2PWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "older driver with an RTX 4000 series GPU" not in record.getMessage()


logging.getLogger("accelerate.big_modeling").addFilter(IgnoreAccelerateP2PWarning())


def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0/true/false/yes/no/on/off, got {value!r}")


def parse_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def normalize_model_size(model_size: str) -> str:
    normalized = model_size.strip().upper().replace("-", "")
    if not normalized.endswith("B"):
        normalized = f"{normalized}B"
    if normalized not in SUPPORTED_QWEN2VL_MODELS:
        supported = ", ".join(SUPPORTED_QWEN2VL_MODELS)
        raise ValueError(f"Unsupported QWEN2VL_MODEL_SIZE={model_size!r}. Choose from: {supported}")
    return normalized


def resolve_model_name(default_model_size: str) -> tuple[str, str]:
    model_name_override = os.environ.get("QWEN2VL_MODEL_NAME")
    if model_name_override:
        resolved_size = os.environ.get("QWEN2VL_MODEL_SIZE", default_model_size)
        return model_name_override, normalize_model_size(resolved_size)

    model_size = normalize_model_size(os.environ.get("QWEN2VL_MODEL_SIZE", default_model_size))
    return SUPPORTED_QWEN2VL_MODELS[model_size], model_size


def slugify_output_tag(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def resolve_output_tag() -> str | None:
    explicit_tag = os.environ.get("QWEN2VL_OUTPUT_TAG", "").strip()
    return slugify_output_tag(explicit_tag) if explicit_tag else None


def resolve_adapter_path() -> Path | None:
    explicit_adapter_path = os.environ.get("QWEN2VL_ADAPTER_PATH", "").strip()
    if explicit_adapter_path:
        return Path(explicit_adapter_path)

    finetune_dir = os.environ.get("QWEN2VL_FINETUNE_DIR", "").strip()
    if not finetune_dir:
        return None

    finetune_path = Path(finetune_dir)
    adapter_dir = finetune_path / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        return adapter_dir
    if (finetune_path / "adapter_config.json").exists():
        return finetune_path
    return adapter_dir


def output_filename_for_model(model_name: str, output_tag: str | None = None) -> str:
    model_slug = model_name.split("/")[-1].lower().replace("-", "_")
    if output_tag:
        return f"{model_slug}_all_results_{output_tag}.json"
    return f"{model_slug}_all_results.json"


def resolve_torch_dtype() -> torch.dtype | str:
    requested_dtype = os.environ.get("QWEN2VL_DTYPE", "").strip().lower()
    if not requested_dtype or requested_dtype == "auto":
        return "auto"
    if requested_dtype == "bf16":
        return torch.bfloat16
    if requested_dtype == "fp16":
        return torch.float16
    raise ValueError("QWEN2VL_DTYPE must be 'auto', 'bf16', or 'fp16'.")


def build_runtime_settings(defaults: ProjectDefaults) -> RuntimeSettings:
    model_name, model_size = resolve_model_name(defaults.default_model_size)
    videos_root = Path(os.environ.get("QWEN2VL_VIDEOS_ROOT", str(defaults.videos_root)))
    output_tag = resolve_output_tag()
    output_path = Path(
        os.environ.get(
            "QWEN2VL_OUTPUT_PATH",
            str(defaults.output_dir / output_filename_for_model(model_name, output_tag)),
        )
    )

    settings = RuntimeSettings(
        model_name=model_name,
        model_size=model_size,
        adapter_path=resolve_adapter_path(),
        videos_root=videos_root,
        output_path=output_path,
        exclude_object_ids_path=(
            Path(exclude_path)
            if (exclude_path := os.environ.get("QWEN2VL_EXCLUDE_OBJECT_IDS_PATH", "").strip())
            else None
        ),
        num_sampled_frames=parse_int_env("QWEN2VL_NUM_SAMPLED_FRAMES", defaults.num_sampled_frames),
        frame_max_pixels=parse_int_env("QWEN2VL_FRAME_MAX_PIXELS", defaults.frame_max_pixels),
        batch_size=parse_int_env("QWEN2VL_BATCH_SIZE", defaults.batch_size),
        max_videos=parse_int_env("QWEN2VL_MAX_VIDEOS", defaults.max_videos),
        sample_seed=parse_int_env("QWEN2VL_SAMPLE_SEED", defaults.sample_seed),
        minimum_num_sampled_frames=parse_int_env(
            "QWEN2VL_MINIMUM_NUM_SAMPLED_FRAMES",
            defaults.minimum_num_sampled_frames,
        ),
        max_new_tokens=parse_int_env("QWEN2VL_MAX_NEW_TOKENS", defaults.max_new_tokens),
        device_map=os.environ.get("QWEN2VL_DEVICE_MAP", defaults.device_map).strip(),
        load_in_8bit=parse_bool_env("QWEN2VL_LOAD_IN_8BIT", defaults.load_in_8bit),
        torch_dtype=resolve_torch_dtype(),
    )

    if settings.num_sampled_frames < 1:
        raise ValueError("QWEN2VL_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.frame_max_pixels < 1:
        raise ValueError("QWEN2VL_FRAME_MAX_PIXELS must be >= 1.")
    if settings.minimum_num_sampled_frames < 1:
        raise ValueError("QWEN2VL_MINIMUM_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.minimum_num_sampled_frames > settings.num_sampled_frames:
        raise ValueError("Minimum sampled frames cannot exceed total sampled frames.")
    if settings.batch_size < 1:
        raise ValueError("QWEN2VL_BATCH_SIZE must be >= 1.")
    if settings.max_videos < 1:
        raise ValueError("QWEN2VL_MAX_VIDEOS must be >= 1.")
    if settings.max_new_tokens < 1:
        raise ValueError("QWEN2VL_MAX_NEW_TOKENS must be >= 1.")
    if not settings.device_map:
        raise ValueError("QWEN2VL_DEVICE_MAP cannot be empty.")

    return settings


def build_greedy_generation_config(max_new_tokens: int) -> dict:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "typical_p": None,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "num_beams": 1,
        "early_stopping": False,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "length_penalty": 1.0,
    }


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


def build_frame_retry_schedule(initial_num_frames: int, minimum_num_sampled_frames: int) -> list[int]:
    candidate_counts = {
        initial_num_frames,
        max(minimum_num_sampled_frames, initial_num_frames // 2),
        max(minimum_num_sampled_frames, initial_num_frames // 3),
        4,
        3,
        2,
        1,
    }
    return sorted(
        {
            count
            for count in candidate_counts
            if minimum_num_sampled_frames <= count <= initial_num_frames
        },
        reverse=True,
    )


def is_cuda_oom_error(error: Exception) -> bool:
    return "cuda out of memory" in str(error).lower()


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


def build_max_memory() -> dict | None:
    if not torch.cuda.is_available():
        return None

    max_memory = {"cpu": "128GiB"}
    for device_idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        gpu_budget_gib = max(4, int(total_gib - 3))
        max_memory[device_idx] = f"{gpu_budget_gib}GiB"
    return max_memory


def iter_auto_model_classes():
    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            yield model_cls


def load_model(settings: RuntimeSettings):
    load_errors = []
    load_kwargs = {
        "dtype": settings.torch_dtype,
        "trust_remote_code": True,
    }

    if settings.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    if torch.cuda.is_available():
        load_kwargs["device_map"] = settings.device_map
        if settings.device_map == "auto":
            max_memory = build_max_memory()
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
                print(f"Using max_memory={max_memory}")

    for model_cls in iter_auto_model_classes():
        try:
            return model_cls.from_pretrained(
                settings.model_name,
                **load_kwargs,
                # attn_implementation="flash_attention_2",  # enable if available
            )
        except Exception as exc:
            load_errors.append(f"{model_cls.__name__}: {exc}")

    if not load_errors:
        raise RuntimeError(
            "No compatible Transformers auto model class is available. "
            "If you upgraded Transformers for Qwen3-VL, also upgrade PyTorch to >= 2.4."
        )

    joined_errors = "\n".join(load_errors)
    raise RuntimeError(
        "Failed to load the Qwen2-VL model with the installed transformers setup. "
        "Upgrade transformers or use an environment with Qwen2-VL support.\n"
        f"{joined_errors}"
    )


def attach_adapter_if_requested(model, settings: RuntimeSettings):
    if settings.adapter_path is None:
        return model

    if not settings.adapter_path.exists():
        raise FileNotFoundError(f"Qwen2-VL adapter path does not exist: {settings.adapter_path}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading a Qwen2-VL finetuned adapter requires `peft`. "
            "Install it with `python -m pip install peft`."
        ) from exc

    model = PeftModel.from_pretrained(model, str(settings.adapter_path))
    print(f"Loaded Qwen2-VL adapter from {settings.adapter_path}")
    return model


def load_model_and_processor(settings: RuntimeSettings):
    model = attach_adapter_if_requested(load_model(settings), settings).eval()
    processor = AutoProcessor.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
        max_pixels=settings.frame_max_pixels,
    )
    if not hasattr(processor, "image_processor"):
        raise RuntimeError(
            "AutoProcessor did not return an image processor for "
            f"{settings.model_name}. For Qwen3-VL, upgrade Transformers with "
            "`pip install -U git+https://github.com/huggingface/transformers`."
        )
    return model, processor


def resolve_model_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return model_device
    return next(model.parameters()).device


def build_message(num_images: int, prompt: str) -> dict:
    return {
        "role": "user",
        "content": (
            [
                {
                    "type": "image",
                }
                for _ in range(num_images)
            ]
            + [{"type": "text", "text": prompt}]
        ),
    }


def extract_first_json_object(text: str):
    decoder = json.JSONDecoder()
    text = text.strip()

    try:
        return json.loads(text), None
    except Exception:
        pass

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return obj, None
        except Exception:
            continue

    return None, "No valid JSON object found"


def batched_inference(
    model,
    processor,
    batch_video_paths: list[Path],
    settings: RuntimeSettings,
    num_frames: int,
) -> list[dict]:
    prompt = build_prompt()

    messages_batch = []
    frames_batch = []
    meta = []

    for video_path in batch_video_paths:
        sampled_frames = sample_video_frames(video_path, num_frames)
        messages = [build_message(len(sampled_frames), prompt)]
        messages_batch.append(messages)
        frames_batch.append(sampled_frames)
        meta.append(
            {
                "video_path": str(video_path),
                "object_id": video_path.stem,
                "video_name": video_path.name,
                "model_name": settings.model_name,
                "model_size": settings.model_size,
                "num_sampled_frames": len(sampled_frames),
            }
        )

    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    inputs = processor(
        text=texts,
        images=frames_batch,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(resolve_model_device(model))

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **build_greedy_generation_config(max_new_tokens=settings.max_new_tokens),
        )

    trimmed_ids = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    decoded = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for item, output_text in zip(meta, decoded):
        parsed, parse_error = extract_first_json_object(output_text)

        results.append(
            {
                **item,
                "raw_output": output_text,
                "parsed_output": parsed,
                "parse_error": parse_error,
            }
        )

    return results


def run_batched_inference_with_retry(
    model,
    processor,
    batch_video_paths: list[Path],
    settings: RuntimeSettings,
) -> list[dict]:
    frame_retry_schedule = build_frame_retry_schedule(
        initial_num_frames=settings.num_sampled_frames,
        minimum_num_sampled_frames=settings.minimum_num_sampled_frames,
    )

    for attempt_idx, current_num_frames in enumerate(frame_retry_schedule):
        try:
            return batched_inference(
                model=model,
                processor=processor,
                batch_video_paths=batch_video_paths,
                settings=settings,
                num_frames=current_num_frames,
            )
        except Exception as error:
            is_last_attempt = attempt_idx == len(frame_retry_schedule) - 1
            if not is_cuda_oom_error(error) or is_last_attempt:
                raise

            next_num_frames = frame_retry_schedule[attempt_idx + 1]
            names = ", ".join(path.name for path in batch_video_paths)
            print(
                f"CUDA OOM for batch [{names}] with {current_num_frames} frames; "
                f"retrying with {next_num_frames} frames."
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError("Frame retry schedule unexpectedly exhausted.")


def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def select_video_paths(settings: RuntimeSettings) -> list[Path]:
    video_paths = sorted(settings.videos_root.glob("*.mp4"))
    if settings.exclude_object_ids_path is not None:
        with settings.exclude_object_ids_path.open("r", encoding="utf-8") as handle:
            excluded = {line.strip() for line in handle if line.strip()}
        video_paths = [path for path in video_paths if path.stem not in excluded]
    if len(video_paths) > settings.max_videos:
        video_paths = sorted(random.Random(settings.sample_seed).sample(video_paths, settings.max_videos))
    return video_paths


def print_runtime_settings(settings: RuntimeSettings) -> None:
    dtype_name = settings.torch_dtype
    if settings.torch_dtype == torch.bfloat16:
        dtype_name = "bf16"
    elif settings.torch_dtype == torch.float16:
        dtype_name = "fp16"

    print("Qwen2-VL inference settings:")
    print(f"  model_name={settings.model_name}")
    print(f"  model_size={settings.model_size}")
    print(f"  adapter_path={settings.adapter_path}")
    print(f"  videos_root={settings.videos_root}")
    print(f"  output_path={settings.output_path}")
    print(f"  num_sampled_frames={settings.num_sampled_frames}")
    print(f"  frame_max_pixels={settings.frame_max_pixels}")
    print(f"  batch_size={settings.batch_size}")
    print(f"  max_videos={settings.max_videos}")
    print(f"  sample_seed={settings.sample_seed}")
    print(f"  minimum_num_sampled_frames={settings.minimum_num_sampled_frames}")
    print(f"  max_new_tokens={settings.max_new_tokens}")
    print(f"  device_map={settings.device_map}")
    print(f"  load_in_8bit={settings.load_in_8bit}")
    print(f"  torch_dtype={dtype_name}")


def main(defaults: ProjectDefaults) -> None:
    settings = build_runtime_settings(defaults)
    print_runtime_settings(settings)

    video_paths = select_video_paths(settings)
    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {settings.videos_root}")

    print(f"Selected {len(video_paths)} videos.")
    model, processor = load_model_and_processor(settings)
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, settings.batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        try:
            batch_results = run_batched_inference_with_retry(model, processor, batch_video_paths, settings)
            all_results.extend(batch_results)

            for result in batch_results:
                print(f"Processed: {result['video_path']}")
                print(result["raw_output"])

        except Exception as error:
            for video_path in batch_video_paths:
                all_results.append(
                    {
                        "video_path": str(video_path),
                        "object_id": video_path.stem,
                        "video_name": video_path.name,
                        "model_name": settings.model_name,
                        "model_size": settings.model_size,
                        "error": str(error),
                    }
                )
            print(f"Batch failed: {error}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(settings.output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {settings.output_path}")


def run_all_projects() -> None:
    failures = []
    for project_name in PROJECT_NAMES:
        videos_root = project_videos_root(project_name)
        output_dir = project_root(project_name)
        print(f"\n=== Qwen2-VL inference: {project_name} ===")
        defaults = ProjectDefaults(
            videos_root=videos_root,
            output_dir=output_dir,
        )
        try:
            main(defaults)
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"Qwen2-VL inference failed for: {failed}")


if __name__ == "__main__":
    run_all_projects()
