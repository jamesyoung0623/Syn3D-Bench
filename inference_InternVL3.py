import gc
import json
import logging
import math
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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel, AutoTokenizer

from inference_common import PROJECT_NAMES, project_root, project_videos_root, setup_run_logging


setup_run_logging("internvl3", __file__)


SUPPORTED_INTERNVL3_MODELS = {
    "1B": "OpenGVLab/InternVL3_5-1B-Instruct",
    "2B": "OpenGVLab/InternVL3_5-2B-Instruct",
    "4B": "OpenGVLab/InternVL3_5-4B-Instruct",
    "8B": "OpenGVLab/InternVL3_5-8B-Instruct",
    "14B": "OpenGVLab/InternVL3_5-14B-Instruct",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
    category=FutureWarning,
)


@dataclass(frozen=True)
class ProjectDefaults:
    videos_root: Path
    output_dir: Path
    default_model_size: str = "8B"
    num_sampled_frames: int = 12
    batch_size: int = 1
    max_videos: int = 1000
    sample_seed: int = 0
    image_size: int = 448
    minimum_num_sampled_frames: int = 1
    max_tiles_per_frame: int = 1
    use_thumbnail: bool = True
    max_new_tokens: int = 128
    device_map_mode: str = "auto"
    load_in_8bit: bool = False


@dataclass(frozen=True)
class RuntimeSettings:
    model_name: str
    model_size: str
    videos_root: Path
    output_path: Path
    exclude_object_ids_path: Path | None
    num_sampled_frames: int
    batch_size: int
    max_videos: int
    sample_seed: int
    image_size: int
    minimum_num_sampled_frames: int
    max_tiles_per_frame: int
    use_thumbnail: bool
    max_new_tokens: int
    device_map_mode: str
    load_in_8bit: bool
    torch_dtype: torch.dtype


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
    if normalized not in SUPPORTED_INTERNVL3_MODELS:
        supported = ", ".join(SUPPORTED_INTERNVL3_MODELS)
        raise ValueError(f"Unsupported INTERNVL3_MODEL_SIZE={model_size!r}. Choose from: {supported}")
    return normalized


def resolve_model_name(default_model_size: str) -> tuple[str, str]:
    model_name_override = os.environ.get("INTERNVL3_MODEL_NAME")
    if model_name_override:
        resolved_size = os.environ.get("INTERNVL3_MODEL_SIZE", default_model_size)
        return model_name_override, normalize_model_size(resolved_size)

    model_size = normalize_model_size(os.environ.get("INTERNVL3_MODEL_SIZE", default_model_size))
    return SUPPORTED_INTERNVL3_MODELS[model_size], model_size


def slugify_output_tag(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def resolve_finetune_output_tag() -> str | None:
    explicit_tag = os.environ.get("INTERNVL3_OUTPUT_TAG", "").strip()
    if explicit_tag:
        return slugify_output_tag(explicit_tag)

    finetune_dir = os.environ.get("INTERNVL3_FINETUNE_DIR", "").strip()
    if finetune_dir:
        finetune_path = Path(finetune_dir)
        if finetune_path.name.startswith("checkpoint") and finetune_path.parent.name:
            if finetune_path.name == "checkpoint-final":
                return slugify_output_tag(finetune_path.parent.name)
            return slugify_output_tag(f"{finetune_path.parent.name}_{finetune_path.name}")
        return slugify_output_tag(finetune_path.name)

    for env_name in ("INTERNVL3_PROJECTOR_PATH", "INTERNVL3_LLM_ADAPTER_PATH"):
        artifact_path_raw = os.environ.get(env_name, "").strip()
        if not artifact_path_raw:
            continue
        artifact_path = Path(artifact_path_raw)
        if artifact_path.suffix:
            base_name = artifact_path.parent.name or artifact_path.stem
        else:
            base_name = artifact_path.name
        return slugify_output_tag(base_name)

    return None


def output_filename_for_model(model_name: str, image_size: int, output_tag: str | None = None) -> str:
    model_slug = model_name.split("/")[-1].lower().replace("-", "_")
    if output_tag:
        return f"{model_slug}_all_results_{image_size}_{output_tag}.json"
    return f"{model_slug}_all_results_{image_size}.json"


def resolve_torch_dtype() -> torch.dtype:
    requested_dtype = os.environ.get("INTERNVL3_DTYPE", "").strip().lower()
    if requested_dtype:
        if requested_dtype == "bf16":
            return torch.bfloat16
        if requested_dtype == "fp16":
            return torch.float16
        raise ValueError("INTERNVL3_DTYPE must be 'bf16' or 'fp16'.")

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if torch.cuda.is_available() and callable(bf16_supported) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_runtime_settings(defaults: ProjectDefaults) -> RuntimeSettings:
    model_name, model_size = resolve_model_name(defaults.default_model_size)
    videos_root = Path(os.environ.get("INTERNVL3_VIDEOS_ROOT", str(defaults.videos_root)))
    image_size = parse_int_env("INTERNVL3_IMAGE_SIZE", defaults.image_size)
    output_tag = resolve_finetune_output_tag()
    output_path = Path(
        os.environ.get(
            "INTERNVL3_OUTPUT_PATH",
            str(defaults.output_dir / output_filename_for_model(model_name, image_size, output_tag)),
        )
    )
    device_map_mode = os.environ.get("INTERNVL3_DEVICE_MAP_MODE", defaults.device_map_mode).strip().lower()
    if device_map_mode not in {"auto", "split"}:
        raise ValueError("INTERNVL3_DEVICE_MAP_MODE must be 'auto' or 'split'.")

    settings = RuntimeSettings(
        model_name=model_name,
        model_size=model_size,
        videos_root=videos_root,
        output_path=output_path,
        exclude_object_ids_path=(
            Path(exclude_path)
            if (exclude_path := os.environ.get("INTERNVL3_EXCLUDE_OBJECT_IDS_PATH", "").strip())
            else None
        ),
        num_sampled_frames=parse_int_env("INTERNVL3_NUM_SAMPLED_FRAMES", defaults.num_sampled_frames),
        batch_size=parse_int_env("INTERNVL3_BATCH_SIZE", defaults.batch_size),
        max_videos=parse_int_env("INTERNVL3_MAX_VIDEOS", defaults.max_videos),
        sample_seed=parse_int_env("INTERNVL3_SAMPLE_SEED", defaults.sample_seed),
        image_size=image_size,
        minimum_num_sampled_frames=parse_int_env(
            "INTERNVL3_MINIMUM_NUM_SAMPLED_FRAMES",
            defaults.minimum_num_sampled_frames,
        ),
        max_tiles_per_frame=parse_int_env("INTERNVL3_MAX_TILES_PER_FRAME", defaults.max_tiles_per_frame),
        use_thumbnail=parse_bool_env("INTERNVL3_USE_THUMBNAIL", defaults.use_thumbnail),
        max_new_tokens=parse_int_env("INTERNVL3_MAX_NEW_TOKENS", defaults.max_new_tokens),
        device_map_mode=device_map_mode,
        load_in_8bit=parse_bool_env("INTERNVL3_LOAD_IN_8BIT", defaults.load_in_8bit),
        torch_dtype=resolve_torch_dtype(),
    )

    if settings.num_sampled_frames < 1:
        raise ValueError("INTERNVL3_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.minimum_num_sampled_frames < 1:
        raise ValueError("INTERNVL3_MINIMUM_NUM_SAMPLED_FRAMES must be >= 1.")
    if settings.minimum_num_sampled_frames > settings.num_sampled_frames:
        raise ValueError("Minimum sampled frames cannot exceed total sampled frames.")
    if settings.max_tiles_per_frame < 1:
        raise ValueError("INTERNVL3_MAX_TILES_PER_FRAME must be >= 1.")

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


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int,
    max_num: int,
    image_size: int,
    use_thumbnail: bool,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    tiles_per_row = target_width // image_size
    for tile_idx in range(blocks):
        box = (
            (tile_idx % tiles_per_row) * image_size,
            (tile_idx // tiles_per_row) * image_size,
            ((tile_idx % tiles_per_row) + 1) * image_size,
            ((tile_idx // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def resolve_num_image_tokens(model, input_size: int) -> int:
    patch_size = model.config.vision_config.patch_size
    downsample_ratio = model.config.downsample_ratio
    if input_size % patch_size != 0:
        raise ValueError(
            f"image_size={input_size} must be divisible by patch_size={patch_size}."
        )
    patches_per_side = input_size // patch_size
    return int((patches_per_side ** 2) * (downsample_ratio ** 2))


def configure_model_for_image_size(model, image_size: int) -> None:
    configured_image_size = getattr(model.config, "force_image_size", None)
    if configured_image_size is None:
        configured_image_size = getattr(model.config.vision_config, "image_size", image_size)

    if image_size == configured_image_size:
        return

    model.num_image_token = resolve_num_image_tokens(model, image_size)
    print(
        f"Adjusted model.num_image_token for image_size={image_size} "
        f"(config image_size={configured_image_size}) -> {model.num_image_token}"
    )


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


def build_max_memory() -> dict | None:
    if not torch.cuda.is_available():
        return None

    max_memory = {"cpu": "128GiB"}
    for device_idx in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        gpu_budget_gib = max(4, int(total_gib - 3))
        max_memory[device_idx] = f"{gpu_budget_gib}GiB"
    return max_memory


def build_split_device_map(model_name: str) -> dict:
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        raise ValueError("Split device map requires at least two GPUs.")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    device_map = {}
    layer_cnt = 0
    for device_idx, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = device_idx
            layer_cnt += 1

    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


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


def normalize_device_spec(device_spec) -> torch.device:
    if isinstance(device_spec, int):
        return torch.device(f"cuda:{device_spec}")
    if isinstance(device_spec, str):
        if device_spec == "cpu":
            return torch.device("cpu")
        if device_spec.startswith("cuda"):
            return torch.device(device_spec)
    return torch.device("cpu")


def resolve_visual_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for module_name in ("vision_model", "mlp1"):
            if module_name in hf_device_map:
                return normalize_device_spec(hf_device_map[module_name])
        for module_name, device_spec in hf_device_map.items():
            if module_name.startswith("vision_model"):
                return normalize_device_spec(device_spec)

    return getattr(model, "device", torch.device("cpu"))


def prepare_internvl_inputs(
    model,
    frames: list[Image.Image],
    settings: RuntimeSettings,
) -> tuple[torch.Tensor, list[int], str]:
    transform = build_transform(settings.image_size)
    pixel_values_list = []
    num_patches_list = []

    for frame in frames:
        tiles = dynamic_preprocess(
            image=frame,
            min_num=1,
            max_num=settings.max_tiles_per_frame,
            image_size=settings.image_size,
            use_thumbnail=settings.use_thumbnail,
        )
        pixel_values = torch.stack([transform(tile) for tile in tiles])
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    visual_device = resolve_visual_device(model)
    pixel_values = torch.cat(pixel_values_list, dim=0).to(
        device=visual_device,
        dtype=settings.torch_dtype,
    )
    frame_placeholders = "\n".join(f"Frame {idx + 1}: <image>" for idx in range(len(frames)))
    question = f"{frame_placeholders}\n\n{build_prompt()}"
    return pixel_values, num_patches_list, question


def run_inference_on_video(model, tokenizer, video_path: Path, settings: RuntimeSettings) -> dict:
    generation_config = build_greedy_generation_config(max_new_tokens=settings.max_new_tokens)
    frame_retry_schedule = build_frame_retry_schedule(
        initial_num_frames=settings.num_sampled_frames,
        minimum_num_sampled_frames=settings.minimum_num_sampled_frames,
    )

    for attempt_idx, current_num_frames in enumerate(frame_retry_schedule):
        sampled_frames = None
        pixel_values = None
        try:
            sampled_frames = sample_video_frames(video_path, current_num_frames)
            pixel_values, num_patches_list, question = prepare_internvl_inputs(
                model=model,
                frames=sampled_frames,
                settings=settings,
            )

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

            parsed = None
            parse_error = None
            try:
                parsed = json.loads(response)
            except Exception as error:
                parse_error = str(error)

            return {
                "video_path": str(video_path),
                "object_id": video_path.stem,
                "video_name": video_path.name,
                "model_name": settings.model_name,
                "model_size": settings.model_size,
                "image_size": settings.image_size,
                "num_sampled_frames": len(sampled_frames),
                "num_tiles": sum(num_patches_list),
                "num_patches_list": num_patches_list,
                "raw_output": response,
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
            del pixel_values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def chunk_list(items, size):
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def select_video_paths(settings: RuntimeSettings) -> list[Path]:
    video_paths = sorted(settings.videos_root.glob("*.mp4"))
    if settings.exclude_object_ids_path is not None:
        with settings.exclude_object_ids_path.open("r", encoding="utf-8") as handle:
            excluded = {line.strip() for line in handle if line.strip()}
        video_paths = [path for path in video_paths if path.stem not in excluded]
    if len(video_paths) > settings.max_videos:
        video_paths = sorted(random.Random(settings.sample_seed).sample(video_paths, settings.max_videos))
    return video_paths


def load_model_and_tokenizer(settings: RuntimeSettings):
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_flash_attn": True,
    }

    if settings.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = settings.torch_dtype

    if torch.cuda.is_available():
        if settings.device_map_mode == "split" and torch.cuda.device_count() > 1:
            device_map = build_split_device_map(settings.model_name)
            load_kwargs["device_map"] = device_map
            print(f"Using split device_map across {torch.cuda.device_count()} GPUs.")
        else:
            max_memory = build_max_memory()
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
                print(f"Using max_memory={max_memory}")
            load_kwargs["device_map"] = "auto"

    model = AutoModel.from_pretrained(settings.model_name, **load_kwargs).eval()
    configure_model_for_image_size(model, settings.image_size)
    maybe_load_finetuned_weights(model)
    return model, tokenizer


def maybe_load_finetuned_weights(model) -> None:
    finetune_dir = os.environ.get("INTERNVL3_FINETUNE_DIR")
    projector_path = os.environ.get("INTERNVL3_PROJECTOR_PATH")
    llm_adapter_path = os.environ.get("INTERNVL3_LLM_ADAPTER_PATH")

    if finetune_dir:
        finetune_dir_path = Path(finetune_dir)
        if projector_path is None:
            candidate = finetune_dir_path / "projector.pt"
            if candidate.exists():
                projector_path = str(candidate)
        if llm_adapter_path is None:
            candidate = finetune_dir_path / "llm_adapter"
            if candidate.exists():
                llm_adapter_path = str(candidate)

    if projector_path:
        projector_state = torch.load(projector_path, map_location="cpu")
        model.mlp1.load_state_dict(projector_state, strict=True)
        print(f"Loaded finetuned projector from {projector_path}")

    if llm_adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "Loading INTERNVL3_LLM_ADAPTER_PATH requires the `peft` package."
            ) from exc

        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            llm_adapter_path,
            is_trainable=False,
        )
        model.language_model.eval()
        print(f"Loaded LLM adapter from {llm_adapter_path}")


def print_runtime_settings(settings: RuntimeSettings) -> None:
    dtype_name = "bf16" if settings.torch_dtype == torch.bfloat16 else "fp16"
    print("InternVL3 inference settings:")
    print(f"  model_name={settings.model_name}")
    print(f"  model_size={settings.model_size}")
    print(f"  videos_root={settings.videos_root}")
    print(f"  output_path={settings.output_path}")
    print(f"  image_size={settings.image_size}")
    print(f"  num_sampled_frames={settings.num_sampled_frames}")
    print(f"  max_tiles_per_frame={settings.max_tiles_per_frame}")
    print(f"  use_thumbnail={settings.use_thumbnail}")
    print(f"  device_map_mode={settings.device_map_mode}")
    print(f"  load_in_8bit={settings.load_in_8bit}")
    print(f"  torch_dtype={dtype_name}")


def main(defaults: ProjectDefaults) -> None:
    settings = build_runtime_settings(defaults)
    print_runtime_settings(settings)

    video_paths = select_video_paths(settings)
    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {settings.videos_root}")

    print(f"Selected {len(video_paths)} videos.")
    model, tokenizer = load_model_and_tokenizer(settings)
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, settings.batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(model, tokenizer, video_path, settings)
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
                    "image_size": settings.image_size,
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
    failures = []
    for project_name in PROJECT_NAMES:
        videos_root = project_videos_root(project_name)
        output_dir = project_root(project_name)
        print(f"\n=== InternVL3 inference: {project_name} ===")
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
        raise RuntimeError(f"InternVL3 inference failed for: {failed}")


if __name__ == "__main__":
    run_all_projects()
