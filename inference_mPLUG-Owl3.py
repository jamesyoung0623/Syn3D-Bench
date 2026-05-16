import gc
import json
import logging
import os
from pathlib import Path
import random
import traceback
import warnings

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_common import iter_project_configs

model_name = os.environ.get("MPLUG_OWL3_MODEL", "mPLUG/mPLUG-Owl3-7B-241101")
output_filename = "mplug_owl3_7b_all_results.json"
videos_root = Path()
output_path = Path()

num_sampled_frames = 12
minimum_num_sampled_frames = 1
batch_size = 1
max_new_tokens = 128
max_videos = 1000
sample_seed = 0
attn_implementation = os.environ.get("MPLUG_OWL3_ATTN_IMPLEMENTATION", "sdpa").strip() or None

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
    category=FutureWarning,
)


class IgnoreAccelerateP2PWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "older driver with an RTX 4000 series GPU" not in record.getMessage()


logging.getLogger("accelerate.big_modeling").addFilter(IgnoreAccelerateP2PWarning())


def greedy_generation_kwargs(max_new_tokens: int) -> dict:
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


def build_frame_retry_schedule(initial_num_frames: int) -> list[int]:
    candidate_counts = {
        initial_num_frames,
        max(minimum_num_sampled_frames, initial_num_frames // 2),
        max(minimum_num_sampled_frames, initial_num_frames // 3),
        8,
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
    message = str(error).lower()
    return "cuda out of memory" in message or "cublas_status_alloc_failed" in message


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def build_messages(num_images: int, prompt: str) -> list[dict]:
    image_tokens = "\n".join("<|image|>" for _ in range(num_images))
    user_content = f"{image_tokens}\n{prompt}" if image_tokens else prompt
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": ""},
    ]


def get_model_input_device(model) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None and model_device.type != "meta":
        return model_device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_generation_device_alignment(model) -> None:
    language_model = getattr(model, "language_model", None)
    device_map = getattr(model, "hf_device_map", None) or {}
    mapped_devices = {str(device) for device in device_map.values() if str(device) != "disk"}
    if language_model is None or len(mapped_devices) <= 1:
        return

    hf_hook = getattr(language_model, "_hf_hook", None)
    if hf_hook is not None and hasattr(hf_hook, "io_same_device"):
        hf_hook.io_same_device = True
        return

    try:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

        add_hook_to_module(language_model, AlignDevicesHook(io_same_device=True), append=True)
    except Exception:
        pass


def load_model_and_processor(model_name: str):
    load_kwargs = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load mPLUG-Owl3 model `{model_name}`. "
            "Make sure the active environment can load Hugging Face custom-code models "
            "and includes the model-side dependencies required by mPLUG-Owl3."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        processor = model.init_processor(tokenizer)
    except Exception as exc:
        raise RuntimeError(
            f"Loaded `{model_name}` but failed to initialize its tokenizer/processor."
        ) from exc

    configure_generation_device_alignment(model)
    model.eval()
    return model, tokenizer, processor


def extract_first_json_object(text: str):
    decoder = json.JSONDecoder()
    stripped = text.strip()

    try:
        return json.loads(stripped), None
    except Exception:
        pass

    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
            return parsed, None
        except Exception:
            continue

    return None, "No valid JSON object found"


def normalize_generated_text(generated, tokenizer) -> str:
    if isinstance(generated, str):
        return generated.strip()

    if isinstance(generated, (list, tuple)):
        if not generated:
            return ""
        first_item = generated[0]
        if isinstance(first_item, str):
            return first_item.strip()
        if torch.is_tensor(first_item):
            return tokenizer.decode(first_item, skip_special_tokens=True).strip()
        return str(first_item).strip()

    if torch.is_tensor(generated):
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    return str(generated).strip()


def run_inference_with_num_frames(model, tokenizer, processor, video_path: Path, num_frames: int) -> dict:
    sampled_frames = sample_video_frames(video_path, num_frames)
    prompt = build_prompt()
    messages = build_messages(len(sampled_frames), prompt)

    inputs = processor(messages, images=sampled_frames, videos=None)
    inputs = inputs.to(get_model_input_device(model))
    inputs.update(
        {
            "tokenizer": tokenizer,
            "decode_text": True,
            **greedy_generation_kwargs(max_new_tokens=max_new_tokens),
        }
    )

    with torch.no_grad():
        generated = model.generate(**inputs)

    output_text = normalize_generated_text(generated, tokenizer)
    parsed, parse_error = extract_first_json_object(output_text)

    return {
        "video_path": str(video_path),
        "object_id": video_path.stem,
        "video_name": video_path.name,
        "num_sampled_frames": len(sampled_frames),
        "raw_output": output_text,
        "parsed_output": parsed,
        "parse_error": parse_error,
    }


def run_inference_on_video(model, tokenizer, processor, video_path: Path) -> dict:
    last_oom_error = None

    for frame_count in build_frame_retry_schedule(num_sampled_frames):
        try:
            return run_inference_with_num_frames(model, tokenizer, processor, video_path, frame_count)
        except RuntimeError as exc:
            if not is_cuda_oom_error(exc):
                raise
            last_oom_error = exc
            print(
                f"CUDA OOM while processing {video_path.name} with {frame_count} frames. "
                "Retrying with fewer frames."
            )
            clear_memory()

    if last_oom_error is not None:
        raise RuntimeError(
            f"Failed to process {video_path.name} due to CUDA OOM even after reducing frames."
        ) from last_oom_error

    raise RuntimeError(f"Failed to process {video_path.name} for an unknown reason.")


def chunk_list(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


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
    print(f"Using model: {model_name}")
    if attn_implementation is not None:
        print(f"Attention implementation: {attn_implementation}")

    model, tokenizer, processor = load_model_and_processor(model_name)
    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(model, tokenizer, processor, video_path)
                all_results.append(result)

                print("Raw output:")
                print(result["raw_output"])
                if result["parsed_output"] is not None:
                    print("Parsed JSON:")
                    print(result["parsed_output"])
                else:
                    print(f"Model output was not strict JSON. Error: {result['parse_error']}")

            except Exception as exc:
                error_traceback = traceback.format_exc()
                error_result = {
                    "video_path": str(video_path),
                    "object_id": video_path.stem,
                    "video_name": video_path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": error_traceback,
                }
                all_results.append(error_result)
                print(f"Failed: {exc}")
                print(error_traceback)
            finally:
                clear_memory()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    global videos_root, output_path

    failures = []
    for project_name, dataset_videos_root, dataset_output_path in iter_project_configs(output_filename):
        print(f"\n=== mPLUG-Owl3 inference: {project_name} ===")
        videos_root = dataset_videos_root
        output_path = dataset_output_path
        try:
            run_current_dataset()
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"mPLUG-Owl3 inference failed for: {failed}")


if __name__ == "__main__":
    main()
