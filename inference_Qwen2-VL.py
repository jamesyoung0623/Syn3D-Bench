import json
import logging
import os
from pathlib import Path
import random
import warnings

import cv2
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from inference_common import iter_project_configs

model_name = "Qwen/Qwen2-VL-7B-Instruct"
# model_name = "Qwen/Qwen2-VL-72B-Instruct"
output_filename = "qwen2vl_7b_all_results.json"
videos_root = Path()
output_path = Path()

num_sampled_frames = 12
frame_max_pixels = 256 * 28 * 28
batch_size = 1
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


def load_model(model_name: str):
    load_errors = []

    for model_cls in (AutoModelForVision2Seq, AutoModelForCausalLM):
        try:
            return model_cls.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",  # enable if available
            )
        except Exception as exc:
            load_errors.append(f"{model_cls.__name__}: {exc}")

    joined_errors = "\n".join(load_errors)
    raise RuntimeError(
        "Failed to load the Qwen2-VL model with the installed transformers setup. "
        "Upgrade transformers or use an environment with Qwen2-VL support.\n"
        f"{joined_errors}"
    )


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


def batched_inference(model, processor, batch_video_paths: list[Path]) -> list[dict]:
    prompt = build_prompt()

    messages_batch = []
    frames_batch = []
    meta = []

    for video_path in batch_video_paths:
        sampled_frames = sample_video_frames(video_path, num_sampled_frames)
        messages = [build_message(len(sampled_frames), prompt)]
        messages_batch.append(messages)
        frames_batch.append(sampled_frames)
        meta.append(
            {
                "video_path": str(video_path),
                "object_id": video_path.stem,
                "video_name": video_path.name,
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
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **greedy_generation_kwargs(max_new_tokens=256),
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
        parsed = None
        parse_error = None
        try:
            parsed = json.loads(output_text)
        except Exception as e:
            parse_error = str(e)

        results.append(
            {
                **item,
                "raw_output": output_text,
                "parsed_output": parsed,
                "parse_error": parse_error,
            }
        )

    return results


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

    model = load_model(model_name)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        max_pixels=frame_max_pixels,
    )

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        try:
            batch_results = batched_inference(model, processor, batch_video_paths)
            all_results.extend(batch_results)

            for result in batch_results:
                print(f"Processed: {result['video_path']}")
                print(result["raw_output"])

        except Exception as e:
            for video_path in batch_video_paths:
                all_results.append(
                    {
                        "video_path": str(video_path),
                        "object_id": video_path.stem,
                        "video_name": video_path.name,
                        "error": str(e),
                    }
                )
            print(f"Batch failed: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    global videos_root, output_path

    failures = []
    for project_name, dataset_videos_root, dataset_output_path in iter_project_configs(output_filename):
        print(f"\n=== Qwen2-VL inference: {project_name} ===")
        videos_root = dataset_videos_root
        output_path = dataset_output_path
        try:
            run_current_dataset()
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"Qwen2-VL inference failed for: {failed}")


if __name__ == "__main__":
    main()
