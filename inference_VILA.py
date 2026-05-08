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
from PIL import Image
from inference_common import iter_project_configs


def _candidate_vila_roots() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = []

    for env_var in ("VILA_REPO", "VILA_ROOT"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())

    candidates.extend(
        [
            script_dir / "VILA",
            script_dir.parent / "VILA",
            script_dir / "external" / "VILA",
        ]
    )
    return candidates


def _load_vila_imports():
    searched_paths = []

    for candidate_root in _candidate_vila_roots():
        resolved_root = candidate_root.resolve()
        if resolved_root in searched_paths:
            continue
        searched_paths.append(resolved_root)

        if (resolved_root / "llava").is_dir():
            resolved_root_str = str(resolved_root)
            if resolved_root_str not in sys.path:
                sys.path.insert(0, resolved_root_str)

    try:
        from llava.conversation import auto_set_conversation_mode
        from llava.conversation import CONVERSATION_MODE_MAPPING
        from llava.conversation import SeparatorStyle
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path
        from llava.mm_utils import is_gemma_tokenizer
        from llava.mm_utils import KeywordsStoppingCriteria
        from llava.mm_utils import process_images
        from llava.mm_utils import tokenizer_image_token
        from llava.model.builder import load_pretrained_model
    except ModuleNotFoundError as exc:
        searched_summary = ", ".join(str(path) for path in searched_paths) or "(none)"
        if exc.name == "llava":
            detail = (
                "The NVLabs VILA codebase is not importable in the active Python environment."
            )
        else:
            detail = (
                f"Found or reached VILA code, but dependency `{exc.name}` is missing "
                "from the active Python environment."
            )

        raise ModuleNotFoundError(
            f"{detail}\n"
            f"Active Python: {sys.executable}\n"
            "This script expects the official NVLabs/VILA checkout, which provides the `llava` package.\n"
            "Fix options:\n"
            "  1. Clone and install VILA into this environment.\n"
            "     git clone https://github.com/NVlabs/VILA.git /path/to/VILA\n"
            "     cd /path/to/VILA && ./environment_setup.sh vila\n"
            "  2. If VILA is already cloned locally, expose it with either:\n"
            "     export VILA_REPO=/path/to/VILA\n"
            "     export PYTHONPATH=/path/to/VILA:$PYTHONPATH\n"
            f"Searched local candidates: {searched_summary}"
        ) from exc

    return (
        load_pretrained_model,
        get_model_name_from_path,
        conv_templates,
        CONVERSATION_MODE_MAPPING,
        auto_set_conversation_mode,
        SeparatorStyle,
        tokenizer_image_token,
        process_images,
        KeywordsStoppingCriteria,
        is_gemma_tokenizer,
    )


(
    load_pretrained_model,
    get_model_name_from_path,
    conv_templates,
    CONVERSATION_MODE_MAPPING,
    auto_set_conversation_mode,
    SeparatorStyle,
    tokenizer_image_token,
    process_images,
    KeywordsStoppingCriteria,
    is_gemma_tokenizer,
) = _load_vila_imports()


model_path = "Efficient-Large-Model/VILA1.5-13b"
# model_path = "Efficient-Large-Model/VILA1.5-40b"
output_filename = "vila15_13b_all_results.json"
videos_root = Path()
output_path = Path()

num_sampled_frames = 12
batch_size = 1
max_new_tokens = 256
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


def build_prompt(num_images: int) -> str:
    image_tokens = "\n".join(["<image>"] * num_images)
    return f"""{image_tokens}

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
{{
  "label": "human-created" | "synthetic" | "uncertain",
  "reason": "one-sentence reason"
}}
""".strip()


def select_conversation_template(model_name_or_path: str):
    normalized_model_name = model_name_or_path.lower()

    for name_fragment, template_name in CONVERSATION_MODE_MAPPING.items():
        if name_fragment in normalized_model_name and template_name in conv_templates:
            return conv_templates[template_name].copy()

    for fallback_name in ("llava_v1", "v1", "vicuna_v1", "plain"):
        if fallback_name in conv_templates:
            return conv_templates[fallback_name].copy()

    available_templates = ", ".join(sorted(conv_templates))
    raise KeyError(f"No supported conversation template found. Available templates: {available_templates}")


def generate_with_vila(tokenizer, model, image_processor, context_len, frames, prompt):
    """
    VILA/LLaVA-style multi-image generation helper.
    Depending on the exact installed VILA version, you may need to slightly
    adjust the model.generate call or image preprocessing here.
    """
    conv = select_conversation_template(model_path)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    image_tensors = process_images(frames, image_processor, model.config)
    image_tensors = image_tensors.to(model.device, dtype=torch.float16)
    media = {"image": [image for image in image_tensors]}
    media_config = {"image": {}}

    input_ids = tokenizer_image_token(
        full_prompt,
        tokenizer,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)

    if conv.sep_style == SeparatorStyle.LLAMA_3:
        stop_str = None
        keywords = [conv.sep, conv.sep2]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
    else:
        stop_str = conv.sep2 if conv.sep_style == SeparatorStyle.TWO else conv.sep
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if is_gemma_tokenizer(tokenizer) else None

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            media=media,
            media_config=media_config,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if stop_str is not None and output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]
    output_text = output_text.strip()

    return output_text


def run_inference_on_video(tokenizer, model, image_processor, context_len, video_path: Path, num_frames: int) -> dict:
    sampled_frames = sample_video_frames(video_path, num_frames)
    prompt = build_prompt(len(sampled_frames))

    output_text = generate_with_vila(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        context_len=context_len,
        frames=sampled_frames,
        prompt=prompt,
    )

    parsed = None
    parse_error = None
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        parse_error = str(e)

    return {
        "video_path": str(video_path),
        "object_id": video_path.stem,
        "video_name": video_path.name,
        "num_sampled_frames": len(sampled_frames),
        "raw_output": output_text,
        "parsed_output": parsed,
        "parse_error": parse_error,
    }


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

    auto_set_conversation_mode(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map="auto",
    )
    model.eval()
    target_num_frames = getattr(model.config, "num_video_frames", None) or num_sampled_frames
    print(f"Using {target_num_frames} frames per video.")

    all_results = []

    for batch_idx, batch_video_paths in enumerate(chunk_list(video_paths, batch_size), start=1):
        print(f"\nBatch {batch_idx}: {len(batch_video_paths)} videos")

        for video_path in batch_video_paths:
            print(f"Processing: {video_path}")
            try:
                result = run_inference_on_video(
                    tokenizer=tokenizer,
                    model=model,
                    image_processor=image_processor,
                    context_len=context_len,
                    video_path=video_path,
                    num_frames=target_num_frames,
                )
                all_results.append(result)

                print("Raw output:")
                print(result["raw_output"])
                if result["parsed_output"] is not None:
                    print("Parsed JSON:")
                    print(result["parsed_output"])
                else:
                    print(f"Model output was not strict JSON. Error: {result['parse_error']}")

            except Exception as e:
                error_traceback = traceback.format_exc()
                error_result = {
                    "video_path": str(video_path),
                    "object_id": video_path.stem,
                    "video_name": video_path.name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": error_traceback,
                }
                all_results.append(error_result)
                print(f"Failed: {e}")
                print(error_traceback)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    global videos_root, output_path

    failures = []
    for project_name, dataset_videos_root, dataset_output_path in iter_project_configs(output_filename):
        print(f"\n=== VILA inference: {project_name} ===")
        videos_root = dataset_videos_root
        output_path = dataset_output_path
        try:
            run_current_dataset()
        except Exception as exc:
            failures.append((project_name, exc))
            print(f"Failed {project_name}: {exc}")

    if failures:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"VILA inference failed for: {failed}")


if __name__ == "__main__":
    main()
