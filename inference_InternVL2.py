import json
import gc
import logging
import os
from pathlib import Path
import random
import sys
import warnings

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_common import iter_project_configs, setup_run_logging

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
setup_run_logging("internvl2", __file__)

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

model_name = "OpenGVLab/InternVL2-26B"
output_filename = "internvl2_26b_all_results_448.json"
videos_root = Path()
output_path = Path()

num_sampled_frames = 12
batch_size = 1
max_videos = 1000
sample_seed = 0
image_size = 448
minimum_num_sampled_frames = 1
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
    category=FutureWarning,
)


class IgnoreAccelerateP2PWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "older driver with an RTX 4000 series GPU" not in record.getMessage()


logging.getLogger("accelerate.big_modeling").addFilter(IgnoreAccelerateP2PWarning())


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


def build_transform(input_size: int = image_size):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def resolve_num_image_tokens(model, input_size: int) -> int:
    patch_size = model.config.vision_config.patch_size
    downsample_ratio = model.config.downsample_ratio
    if input_size % patch_size != 0:
        raise ValueError(
            f"image_size={input_size} must be divisible by patch_size={patch_size}."
        )
    patches_per_side = input_size // patch_size
    return int((patches_per_side ** 2) * (downsample_ratio ** 2))


def configure_model_for_image_size(model) -> None:
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


def build_frame_retry_schedule(initial_num_frames: int) -> list[int]:
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


def build_prompt() -> str:
    return """
You are analyzing multiple rendered images of the same 3D asset.

Your goal is to infer the origin of the underlying 3D model, not to describe the rendered views.

Decide whether the underlying 3D model is:
- "human-created"
- "synthetic"
- "uncertain"

Definitions:

- "human-created":
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
5. For "human-created", the reason should point to evidence of deliberate manual design, functional structure, meaningful detail placement, or coherent asset construction.
6. For "synthetic", the reason should point to evidence of generative artifacts, implausible geometry, repeated or nonsensical structure, over-smoothing, inconsistent semantics, or missing/merged functional parts.
7. For "uncertain", the reason should explain exactly why the visible evidence is not diagnostic.

Output requirements:
- Return valid JSON only.
- Provide exactly one reason in one sentence.
- The reason must cite a diagnostic visual cue.
- Do not repeat the label wording in the reason.

Return JSON with this schema:
{
  "label": "human-created" | "synthetic" | "uncertain",
  "reason": "one-sentence reason"
}
""".strip()


def prepare_internvl_inputs(model, frames: list[Image.Image]) -> tuple[torch.Tensor, list[int], str]:
    transform = build_transform()
    pixel_values = torch.stack([transform(frame) for frame in frames])
    device = getattr(model, "device", torch.device("cpu"))
    dtype = getattr(model, "dtype", torch.float16)
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    num_patches_list = [1] * len(frames)
    frame_placeholders = "\n".join(f"Frame {idx + 1}: <image>" for idx in range(len(frames)))
    question = f"{frame_placeholders}\n\n{build_prompt()}"
    return pixel_values, num_patches_list, question


def run_inference_on_video(model, tokenizer, video_path: Path) -> dict:
    generation_config = build_greedy_generation_config(max_new_tokens=256)
    frame_retry_schedule = build_frame_retry_schedule(num_sampled_frames)

    for attempt_idx, current_num_frames in enumerate(frame_retry_schedule):
        sampled_frames = None
        pixel_values = None
        try:
            sampled_frames = sample_video_frames(video_path, current_num_frames)
            pixel_values, num_patches_list, question = prepare_internvl_inputs(model, sampled_frames)

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
            except Exception as e:
                parse_error = str(e)

            return {
                "video_path": str(video_path),
                "object_id": video_path.stem,
                "video_name": video_path.name,
                "num_sampled_frames": len(sampled_frames),
                "raw_output": response,
                "parsed_output": parsed,
                "parse_error": parse_error,
            }
        except Exception as e:
            is_last_attempt = attempt_idx == len(frame_retry_schedule) - 1
            if not is_cuda_oom_error(e) or is_last_attempt:
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
    for i in range(0, len(items), size):
        yield items[i:i + size]


def select_video_paths() -> list[Path]:
    video_paths = sorted(videos_root.glob("*.mp4"))
    if len(video_paths) > max_videos:
        video_paths = sorted(random.Random(sample_seed).sample(video_paths, max_videos))
    return video_paths


def run_current_dataset():
    video_paths = select_video_paths()
    if not video_paths:
        raise FileNotFoundError(f"No videos found under: {videos_root}")

    print(f"Selected {len(video_paths)} videos.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    max_memory = build_max_memory()
    if max_memory is not None:
        print(f"Using max_memory={max_memory}")

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        device_map="auto",
        max_memory=max_memory,
    ).eval()
    configure_model_for_image_size(model)

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(model, tokenizer, video_path)
                all_results.append(result)

                print("Raw output:")
                print(result["raw_output"])
                if result["parsed_output"] is not None:
                    print("Parsed JSON:")
                    print(result["parsed_output"])
                else:
                    print(f"Model output was not strict JSON. Error: {result['parse_error']}")

            except Exception as e:
                error_result = {
                    "video_path": str(video_path),
                    "object_id": video_path.stem,
                    "video_name": video_path.name,
                    "error": str(e),
                }
                all_results.append(error_result)
                print(f"Failed: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    global videos_root, output_path

    failures = []
    for project_name, dataset_videos_root, dataset_output_path in iter_project_configs(output_filename):
        print(f"\n=== InternVL2 inference: {project_name} ===")
        videos_root = dataset_videos_root
        output_path = dataset_output_path
        try:
            run_current_dataset()
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"InternVL2 inference failed for: {failed}")


if __name__ == "__main__":
    main()
