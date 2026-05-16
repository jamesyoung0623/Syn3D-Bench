import gc
import json
import logging
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from inference_common import PROJECT_NAMES, project_root, project_videos_root, setup_run_logging


setup_run_logging("llava", __file__)


SUPPORTED_LLAVA_MODELS = {
    "4B": "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
    "8B": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
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
    default_model_size: str = "4B"
    num_sampled_frames: int = 12
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
    batch_size: int
    max_videos: int
    sample_seed: int
    minimum_num_sampled_frames: int
    max_new_tokens: int
    device_map: str
    load_in_8bit: bool
    torch_dtype: torch.dtype


class IgnoreAccelerateP2PWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "older driver with an RTX 4000 series GPU" not in record.getMessage()


logging.getLogger("accelerate.big_modeling").addFilter(IgnoreAccelerateP2PWarning())


def patch_transformers_remote_code_compatibility() -> None:
    try:
        import transformers.configuration_utils as configuration_utils

        if not hasattr(configuration_utils, "layer_type_validation"):
            configuration_utils.layer_type_validation = lambda layer_types: layer_types
    except Exception:
        pass

    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils

        if not hasattr(flash_attention_utils, "flash_attn_varlen_func"):
            try:
                from flash_attn import flash_attn_varlen_func

                flash_attention_utils.flash_attn_varlen_func = flash_attn_varlen_func
            except Exception:
                pass

        if not hasattr(flash_attention_utils, "flash_attn_supports_top_left_mask"):
            flash_attention_utils.flash_attn_supports_top_left_mask = lambda: False
    except Exception:
        pass


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
    if normalized not in SUPPORTED_LLAVA_MODELS:
        supported = ", ".join(SUPPORTED_LLAVA_MODELS)
        raise ValueError(f"Unsupported LLAVA_MODEL_SIZE={model_size!r}. Choose from: {supported}")
    return normalized


def resolve_model_name(default_model_size: str) -> tuple[str, str]:
    model_name_override = os.environ.get("LLAVA_MODEL_NAME")
    if model_name_override:
        resolved_size = os.environ.get("LLAVA_MODEL_SIZE", default_model_size)
        return model_name_override, normalize_model_size(resolved_size)

    model_size = normalize_model_size(os.environ.get("LLAVA_MODEL_SIZE", default_model_size))
    return SUPPORTED_LLAVA_MODELS[model_size], model_size


def slugify_output_tag(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def resolve_output_tag() -> str | None:
    explicit_tag = os.environ.get("LLAVA_OUTPUT_TAG", "").strip()
    return slugify_output_tag(explicit_tag) if explicit_tag else None


def resolve_adapter_path() -> Path | None:
    explicit_adapter_path = os.environ.get("LLAVA_ADAPTER_PATH", "").strip()
    if explicit_adapter_path:
        return Path(explicit_adapter_path)

    finetune_dir = os.environ.get("LLAVA_FINETUNE_DIR", "").strip()
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


def resolve_torch_dtype() -> torch.dtype:
    requested_dtype = os.environ.get("LLAVA_DTYPE", "").strip().lower()
    if requested_dtype:
        if requested_dtype == "bf16":
            return torch.bfloat16
        if requested_dtype == "fp16":
            return torch.float16
        raise ValueError("LLAVA_DTYPE must be 'bf16' or 'fp16'.")

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if torch.cuda.is_available() and callable(bf16_supported) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_runtime_settings(defaults: ProjectDefaults) -> RuntimeSettings:
    model_name, model_size = resolve_model_name(defaults.default_model_size)
    videos_root = Path(os.environ.get("LLAVA_VIDEOS_ROOT", str(defaults.videos_root)))
    output_tag = resolve_output_tag()
    output_path = Path(
        os.environ.get(
            "LLAVA_OUTPUT_PATH",
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
            if (exclude_path := os.environ.get("LLAVA_EXCLUDE_OBJECT_IDS_PATH", "").strip())
            else None
        ),
        num_sampled_frames=parse_int_env("LLAVA_NUM_SAMPLED_FRAMES", defaults.num_sampled_frames),
        batch_size=parse_int_env("LLAVA_BATCH_SIZE", defaults.batch_size),
        max_videos=parse_int_env("LLAVA_MAX_VIDEOS", defaults.max_videos),
        sample_seed=parse_int_env("LLAVA_SAMPLE_SEED", defaults.sample_seed),
        minimum_num_sampled_frames=parse_int_env(
            "LLAVA_MINIMUM_NUM_SAMPLED_FRAMES",
            defaults.minimum_num_sampled_frames,
        ),
        max_new_tokens=parse_int_env("LLAVA_MAX_NEW_TOKENS", defaults.max_new_tokens),
        device_map=os.environ.get("LLAVA_DEVICE_MAP", defaults.device_map).strip(),
        load_in_8bit=parse_bool_env("LLAVA_LOAD_IN_8BIT", defaults.load_in_8bit),
        torch_dtype=resolve_torch_dtype(),
    )

    if settings.num_sampled_frames < 1:
        raise ValueError("LLAVA_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.minimum_num_sampled_frames < 1:
        raise ValueError("LLAVA_MINIMUM_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.minimum_num_sampled_frames > settings.num_sampled_frames:
        raise ValueError("Minimum sampled frames cannot exceed total sampled frames.")
    if settings.batch_size < 1:
        raise ValueError("LLAVA_BATCH_SIZE must be >= 1.")
    if settings.max_videos < 1:
        raise ValueError("LLAVA_MAX_VIDEOS must be >= 1.")
    if settings.max_new_tokens < 1:
        raise ValueError("LLAVA_MAX_NEW_TOKENS must be >= 1.")
    if not settings.device_map:
        raise ValueError("LLAVA_DEVICE_MAP cannot be empty.")

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


def build_conversation(frames: list[Image.Image], prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": frame} for frame in frames]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]


def apply_chat_template(processor, conversation: list[dict]) -> str:
    try:
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )


def resolve_model_input_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    if isinstance(model_device, str):
        return torch.device(model_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def move_inputs_to_model(inputs, model, torch_dtype: torch.dtype):
    model_device = resolve_model_input_device(model)
    prepared_inputs = {}

    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(model_device)
            if torch.is_floating_point(value):
                value = value.to(dtype=torch_dtype)
        prepared_inputs[key] = value

    return prepared_inputs


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


def run_inference_on_video(model, processor, video_path: Path, settings: RuntimeSettings) -> dict:
    generation_config = build_greedy_generation_config(max_new_tokens=settings.max_new_tokens)
    frame_retry_schedule = build_frame_retry_schedule(
        initial_num_frames=settings.num_sampled_frames,
        minimum_num_sampled_frames=settings.minimum_num_sampled_frames,
    )

    for attempt_idx, current_num_frames in enumerate(frame_retry_schedule):
        sampled_frames = None
        inputs = None
        output_ids = None
        try:
            sampled_frames = sample_video_frames(video_path, current_num_frames)
            conversation = build_conversation(sampled_frames, build_prompt())
            text = apply_chat_template(processor, conversation)

            inputs = processor(
                images=sampled_frames,
                text=[text],
                return_tensors="pt",
                padding=True,
            )
            inputs = move_inputs_to_model(inputs, model, settings.torch_dtype)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    **generation_config,
                )

            prompt_length = inputs["input_ids"].shape[1]
            generated_only = output_ids[:, prompt_length:]

            output_text = processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            parsed, parse_error = extract_first_json_object(output_text)
            return {
                "video_path": str(video_path),
                "object_id": video_path.stem,
                "video_name": video_path.name,
                "model_name": settings.model_name,
                "model_size": settings.model_size,
                "num_sampled_frames": len(sampled_frames),
                "raw_output": output_text,
                "parsed_output": parsed,
                "parse_error": parse_error,
            }
        except Exception as error:
            is_last_attempt = attempt_idx == len(frame_retry_schedule) - 1
            if not is_cuda_oom_error(error) or is_last_attempt:
                raise

            next_num_frames = frame_retry_schedule[attempt_idx + 1]
            print(
                f"CUDA OOM for {video_path.name} with {current_num_frames} frames; "
                f"retrying with {next_num_frames} frames."
            )
        finally:
            del sampled_frames
            del inputs
            del output_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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


def build_max_memory() -> dict | None:
    if not torch.cuda.is_available():
        return None

    max_memory = {"cpu": "128GiB"}
    for device_idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        gpu_budget_gib = max(4, int(total_gib - 3))
        max_memory[device_idx] = f"{gpu_budget_gib}GiB"
    return max_memory


def attach_adapter_if_requested(model, settings: RuntimeSettings):
    if settings.adapter_path is None:
        return model

    if not settings.adapter_path.exists():
        raise FileNotFoundError(f"LLaVA adapter path does not exist: {settings.adapter_path}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading a LLaVA finetuned adapter requires `peft`. "
            "Install it with `python -m pip install peft`."
        ) from exc

    model = PeftModel.from_pretrained(model, str(settings.adapter_path))
    print(f"Loaded LLaVA adapter from {settings.adapter_path}")
    return model


def load_model_and_processor(settings: RuntimeSettings):
    load_kwargs = {
        "torch_dtype": settings.torch_dtype,
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

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
        **load_kwargs,
    )
    model = attach_adapter_if_requested(model, settings).eval()
    processor = AutoProcessor.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
    )
    if not hasattr(processor, "tokenizer") or not hasattr(processor, "image_processor"):
        raise TypeError(
            "Expected AutoProcessor to return a multimodal processor with tokenizer "
            f"and image_processor, got {type(processor).__name__}."
        )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    return model, processor


def print_runtime_settings(settings: RuntimeSettings) -> None:
    dtype_name = "bf16" if settings.torch_dtype == torch.bfloat16 else "fp16"
    print("LLaVA inference settings:")
    print(f"  model_name={settings.model_name}")
    print(f"  model_size={settings.model_size}")
    print(f"  adapter_path={settings.adapter_path}")
    print(f"  videos_root={settings.videos_root}")
    print(f"  output_path={settings.output_path}")
    print(f"  num_sampled_frames={settings.num_sampled_frames}")
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

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(model, processor, video_path, settings)
                all_results.append(result)

                print("Raw output:")
                print(result["raw_output"])
                if result["parsed_output"] is not None:
                    print("Parsed JSON:")
                    print(result["parsed_output"])
                else:
                    print(f"Model output was not strict JSON. Error: {result['parse_error']}")

            except Exception as error:
                error_result = {
                    "video_path": str(video_path),
                    "object_id": video_path.stem,
                    "video_name": video_path.name,
                    "model_name": settings.model_name,
                    "model_size": settings.model_size,
                    "error": str(error),
                }
                all_results.append(error_result)
                print(f"Failed: {error}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(settings.output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {settings.output_path}")


def run_all_projects() -> None:
    patch_transformers_remote_code_compatibility()

    failures = []
    for project_name in PROJECT_NAMES:
        videos_root = project_videos_root(project_name)
        output_dir = project_root(project_name)
        print(f"\n=== LLaVA inference: {project_name} ===")
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
        raise RuntimeError(f"LLaVA inference failed for: {failed}")


if __name__ == "__main__":
    run_all_projects()
