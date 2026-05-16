import json
import logging
import os
from pathlib import Path
import random
import sys
import traceback
import warnings

import cv2
import numpy as np
import torch
from inference_common import iter_project_configs


def _candidate_longva_roots() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = []

    for env_var in ("LONGVA_REPO", "LONGVA_ROOT"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            script_dir / "LongVA",
            script_dir.parent / "LongVA",
            script_dir / "external" / "LongVA",
        ]
    )
    return candidates


def _load_longva_imports():
    searched_paths = []

    for candidate_root in _candidate_longva_roots():
        resolved_root = candidate_root.resolve()
        if resolved_root in searched_paths:
            continue
        searched_paths.append(resolved_root)

        if (resolved_root / "longva").is_dir():
            resolved_root_str = str(resolved_root)
            if resolved_root_str not in sys.path:
                sys.path.insert(0, resolved_root_str)

    try:
        from longva.constants import IMAGE_TOKEN_INDEX
        from longva.mm_utils import tokenizer_image_token
        from longva.model.builder import load_pretrained_model
    except ModuleNotFoundError as exc:
        searched_summary = ", ".join(str(path) for path in searched_paths) or "(none)"
        if exc.name == "longva":
            detail = "The LongVA codebase is not importable in the active Python environment."
        else:
            detail = (
                f"Found or reached LongVA code, but dependency `{exc.name}` is missing "
                "from the active Python environment."
            )

        raise ModuleNotFoundError(
            f"{detail}\n"
            f"Active Python: {sys.executable}\n"
            "This script expects the official EvolvingLMMs-Lab/LongVA checkout or an installed `longva` package.\n"
            "Fix options:\n"
            "  1. Clone and install LongVA following the official repository instructions.\n"
            "     git clone https://github.com/EvolvingLMMs-Lab/LongVA.git /path/to/LongVA\n"
            "     cd /path/to/LongVA && pip install -r requirements.txt && pip install -e \"longva/.[train]\"\n"
            "  2. If LongVA is already cloned locally, expose it with either:\n"
            "     export LONGVA_REPO=/path/to/LongVA\n"
            "     export PYTHONPATH=/path/to/LongVA:$PYTHONPATH\n"
            f"Searched local candidates: {searched_summary}"
        ) from exc

    return load_pretrained_model, tokenizer_image_token, IMAGE_TOKEN_INDEX


(
    load_pretrained_model,
    tokenizer_image_token,
    IMAGE_TOKEN_INDEX,
) = _load_longva_imports()


model_path = "lmms-lab/LongVA-7B"
model_name = "llava_qwen"
output_filename = "longva_7b_all_results.json"
videos_root = Path()
output_path = Path()

num_sampled_frames = 12
batch_size = 1
max_new_tokens = 128
max_videos = 1000
sample_seed = 0

warnings.filterwarnings(
    "ignore",
    message=r"`Qwen2VLRotaryEmbedding` can now be fully parameterized.*",
    category=FutureWarning,
)


class IgnoreAccelerateP2PWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "older driver with an RTX 4000 series GPU" not in record.getMessage()


logging.getLogger("accelerate.big_modeling").addFilter(IgnoreAccelerateP2PWarning())


def greedy_generation_kwargs(max_tokens: int) -> dict:
    return {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": max_tokens,
    }


def sample_video_frames(video_path: Path, num_frames: int) -> np.ndarray:
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
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        return np.stack(frames, axis=0)
    finally:
        capture.release()


def build_task_prompt() -> str:
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


def build_chat_prompt(task_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<image>\n"
        f"{task_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)

    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Could not determine the model device.") from exc


def get_video_tensor_dtype(model_device: torch.device) -> torch.dtype:
    if model_device.type == "cpu":
        return torch.float32
    return torch.float16


def decode_generated_text(tokenizer, output_ids: torch.Tensor, prompt_length: int) -> str:
    generated_ids = output_ids[:, prompt_length:] if output_ids.shape[1] > prompt_length else output_ids
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    if output_text:
        return output_text
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


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


def run_inference_on_video(tokenizer, model, image_processor, video_path: Path) -> dict:
    sampled_frames = sample_video_frames(video_path, num_sampled_frames)
    full_prompt = build_chat_prompt(build_task_prompt())
    model_device = get_model_device(model)

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    video_tensor = image_processor.preprocess(sampled_frames, return_tensors="pt")["pixel_values"]
    video_tensor = video_tensor.to(model_device, dtype=get_video_tensor_dtype(model_device))

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            pad_token_id=tokenizer.eos_token_id,
            **greedy_generation_kwargs(max_new_tokens),
        )

    output_text = decode_generated_text(tokenizer, output_ids, input_ids.shape[1])
    parsed, parse_error = extract_first_json_object(output_text)

    return {
        "video_path": str(video_path),
        "object_id": video_path.stem,
        "video_name": video_path.name,
        "num_sampled_frames": int(sampled_frames.shape[0]),
        "raw_output": output_text,
        "parsed_output": parsed,
        "parse_error": parse_error,
    }


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

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(tokenizer, model, image_processor, video_path)
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    global videos_root, output_path

    failures = []
    for project_name, dataset_videos_root, dataset_output_path in iter_project_configs(output_filename):
        print(f"\n=== LongVA inference: {project_name} ===")
        videos_root = dataset_videos_root
        output_path = dataset_output_path
        try:
            run_current_dataset()
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"LongVA inference failed for: {failed}")


if __name__ == "__main__":
    main()
