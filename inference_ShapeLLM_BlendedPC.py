import argparse
import csv
import json
import os
import random
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SYN3D_ROOT = Path(__file__).resolve().parent
SHAPELLM_ROOT = Path(os.environ.get("SHAPELLM_ROOT", "/home/jamesyoung0623/ShapeLLM")).resolve()
REPO_ROOT = SHAPELLM_ROOT
if str(SHAPELLM_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPELLM_ROOT))


DEFAULT_PROMPT = (
    "This point cloud does not contain real RGB color. All points are colored white only because "
    "the model was trained with colored input. Ignore color completely and judge only from geometry and structure.\n"
    "Is this point cloud likely an original unedited ShapeNet object or an edited object?\n"
    "If it seems edited, which part appears edited?\n"
    "Answer in exactly three lines:\n"
    "is_edited: yes/no/uncertain\n"
    "edited_part: <part name or none>\n"
    "reason: <short explanation>"
)


def detect_default_device() -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def parse_args():
    default_blendedpc_root = Path("/home/jamesyoung0623/BlendedPC")
    default_output_root = SYN3D_ROOT / "evaluation" / "blendedpc_edit_detection_shapellm"

    parser = argparse.ArgumentParser(
        description=(
            "Run ShapeLLM on BlendedPC originals and edited outputs one point cloud at a time, "
            "asking whether each point cloud appears edited and which part seems edited."
        )
    )
    parser.add_argument(
        "--model_name",
        "--model-path",
        "--model_path",
        nargs="+",
        required=True,
        dest="model_name",
        help=(
            "One or more ShapeLLM model paths or Hugging Face ids, for example "
            "`qizekun/ShapeLLM_13B_general_v1.0` or `./checkpoints/llama-vicuna-7b-8k-finetune`."
        ),
    )
    parser.add_argument("--model-base", "--model_base", dest="model_base", type=str, default=None)
    parser.add_argument("--blendedpc_root", type=Path, default=default_blendedpc_root)
    parser.add_argument("--runs_root", type=Path, default=None, help="Optional override for the BlendedPC outputs root.")
    parser.add_argument(
        "--runs_dir",
        action="append",
        default=None,
        help=(
            "Optional relative or absolute run directory to include. Repeat this flag to restrict evaluation "
            "to selected directories such as `outputs/ulip_chair_runs`."
        ),
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--output_root", type=Path, default=default_output_root)
    parser.add_argument("--limit_runs", type=int, default=0, help="Optional limit on the number of BlendedPC run folders to evaluate.")
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--device-map", "--device_map", dest="device_map", type=str, default="auto")
    parser.add_argument("--conv-mode", "--conv_mode", dest="conv_mode", type=str, default="llava_v1")
    parser.add_argument(
        "--sample_points_num",
        type=int,
        default=None,
        help=(
            "Override model.config.sample_points_num. If omitted, the checkpoint config is used; "
            "if missing from the checkpoint, ShapeLLM's 10000-point training default is used."
        ),
    )
    parser.add_argument(
        "--with_color",
        dest="with_color",
        action="store_true",
        default=None,
        help="Force model.config.with_color=True before point preprocessing.",
    )
    parser.add_argument(
        "--no-with_color",
        dest="with_color",
        action="store_false",
        help="Force model.config.with_color=False before point preprocessing.",
    )
    parser.add_argument(
        "--force_white_color",
        dest="force_white_color",
        action="store_true",
        default=True,
        help="Replace or add RGB channels with all-white values. Default: enabled.",
    )
    parser.add_argument(
        "--preserve_color",
        dest="force_white_color",
        action="store_false",
        help="Keep RGB channels from the point-cloud file when present.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--load-8bit", "--load_8bit", dest="load_8bit", action="store_true")
    parser.add_argument("--load-4bit", "--load_4bit", dest="load_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Seed for point sampling and generation. Use a negative value to skip seeding.")
    parser.add_argument(
        "--skip_missing",
        dest="skip_missing",
        action="store_true",
        default=True,
        help="Skip runs with missing metadata/original/output files instead of failing. Default: enabled.",
    )
    parser.add_argument(
        "--no-skip_missing",
        dest="skip_missing",
        action="store_false",
        help="Fail when a BlendedPC run has missing metadata/original/output files.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    return Path(model_name.rstrip("/")).name.replace(":", "_")


def existing_path_or_repo_relative(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    expanded = Path(os.path.expanduser(value))
    if expanded.exists():
        return str(expanded.resolve())
    if not expanded.is_absolute():
        repo_candidate = REPO_ROOT / expanded
        if repo_candidate.exists():
            return str(repo_candidate.resolve())
    return value


def resolve_runs_root(args) -> Path:
    if args.runs_root is not None:
        return args.runs_root.resolve()
    return (args.blendedpc_root / "outputs").resolve()


def collect_run_dirs(args) -> List[Path]:
    runs_root = resolve_runs_root(args)
    if args.runs_dir:
        run_dirs: List[Path] = []
        for item in args.runs_dir:
            path = Path(item)
            if not path.is_absolute():
                candidate = (args.blendedpc_root / item).resolve()
                if candidate.is_dir():
                    path = candidate
                else:
                    path = (runs_root / item).resolve()
            else:
                path = path.resolve()
            run_dirs.append(path)
    else:
        run_dirs = sorted(
            path for path in runs_root.iterdir()
            if path.is_dir() and path.name.endswith("_runs")
        )
    return run_dirs


def build_record(
    *,
    run_family: str,
    run_folder: Path,
    item_type: str,
    point_cloud_path: Path,
    shape_category: str,
    edit_prompt: str,
    expected_edited_part: Optional[str],
    prompt_text: str,
) -> Dict[str, object]:
    expected_is_edited = item_type == "edited"
    ground_truth_label = "edited" if expected_is_edited else "original_shapenet"
    return OrderedDict(
        run_family=run_family,
        run_folder=run_folder.name,
        item_type=item_type,
        point_cloud_path=str(point_cloud_path),
        shape_category=shape_category,
        ground_truth_label=ground_truth_label,
        expected_is_edited=expected_is_edited,
        expected_edited_part=expected_edited_part if expected_is_edited else None,
        edit_prompt=edit_prompt if expected_is_edited else None,
        shapellm_prompt=prompt_text,
    )


def load_records_for_run(run_dir: Path, blendedpc_root: Path, prompt_text: str) -> List[Dict[str, object]]:
    metadata_path = run_dir / "metadata.json"
    output_pc_path = run_dir / "output_pc.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {run_dir}")
    if not output_pc_path.exists():
        raise FileNotFoundError(f"Missing output_pc.npz in {run_dir}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    input_path = metadata.get("input_path")
    if not input_path:
        raise ValueError(f"`input_path` missing from {metadata_path}")
    input_pc_path = (blendedpc_root / input_path).resolve()
    if not input_pc_path.exists():
        raise FileNotFoundError(f"Original input point cloud not found: {input_pc_path}")

    shape_category = metadata.get("shape_category", "")
    edit_prompt = metadata.get("prompt", "")
    expected_edited_part = metadata.get("part")
    run_family = run_dir.parent.name

    return [
        build_record(
            run_family=run_family,
            run_folder=run_dir,
            item_type="original",
            point_cloud_path=input_pc_path,
            shape_category=shape_category,
            edit_prompt=edit_prompt,
            expected_edited_part=expected_edited_part,
            prompt_text=prompt_text,
        ),
        build_record(
            run_family=run_family,
            run_folder=run_dir,
            item_type="edited",
            point_cloud_path=output_pc_path.resolve(),
            shape_category=shape_category,
            edit_prompt=edit_prompt,
            expected_edited_part=expected_edited_part,
            prompt_text=prompt_text,
        ),
    ]


def collect_records(args) -> List[Dict[str, object]]:
    blendedpc_root = args.blendedpc_root.resolve()
    records: List[Dict[str, object]] = []
    run_count = 0
    skipped_runs = 0
    for run_family_dir in collect_run_dirs(args):
        for run_dir in sorted(path for path in run_family_dir.iterdir() if path.is_dir()):
            if args.limit_runs > 0 and run_count >= args.limit_runs:
                if skipped_runs > 0:
                    print(f"[INFO] Skipped {skipped_runs} incomplete or invalid BlendedPC runs.")
                return records
            try:
                records.extend(load_records_for_run(run_dir, blendedpc_root, args.prompt))
                run_count += 1
            except Exception as exc:
                if args.skip_missing:
                    print(f"[skip] {run_dir}: {exc}")
                    skipped_runs += 1
                    continue
                raise
    if skipped_runs > 0:
        print(f"[INFO] Skipped {skipped_runs} incomplete or invalid BlendedPC runs.")
    return records


def write_csv(records: List[Dict[str, object]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def to_json_serializable(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    serializable = []
    for record in records:
        item = {}
        for key, value in record.items():
            if isinstance(value, Path):
                item[key] = str(value)
            else:
                item[key] = value
        serializable.append(item)
    return serializable


def coerce_point_array(value: object, source: Path, key: str) -> np.ndarray:
    point_cloud = np.asarray(value)
    if point_cloud.dtype == object and point_cloud.shape == ():
        point_cloud = np.asarray(point_cloud.item())
    if point_cloud.ndim == 3:
        point_cloud = point_cloud[0]
    if point_cloud.ndim != 2:
        raise ValueError(f"{source}:{key} is not a 2D point array; shape={point_cloud.shape}")
    if point_cloud.shape[0] in {3, 6} and point_cloud.shape[1] not in {3, 6}:
        point_cloud = point_cloud.T
    if point_cloud.shape[1] < 3:
        raise ValueError(f"{source}:{key} has fewer than 3 channels; shape={point_cloud.shape}")
    if point_cloud.shape[1] > 6:
        point_cloud = point_cloud[:, :6]
    elif 3 < point_cloud.shape[1] < 6:
        point_cloud = point_cloud[:, :3]
    return point_cloud.astype(np.float32, copy=False)


def load_npz_point_cloud(path: Path) -> Tuple[np.ndarray, str]:
    preferred_keys = ["coords", "pointcloud", "pred", "points", "xyz", "arr_0"]
    with np.load(path, allow_pickle=True) as data:
        keys = list(data.files)
        for key in preferred_keys + [key for key in keys if key not in preferred_keys]:
            if key not in data:
                continue
            try:
                return coerce_point_array(data[key], path, key), key
            except Exception:
                continue
    raise ValueError(f"Could not find a point-cloud array in {path}")


def load_point_cloud_file(path: Path, force_white_color: bool) -> Tuple[np.ndarray, Dict[str, object]]:
    path = Path(path).resolve()
    if path.suffix.lower() == ".npz":
        point_cloud, source_key = load_npz_point_cloud(path)
    else:
        from llava.mm_utils import load_pts

        point_cloud = coerce_point_array(load_pts(str(path)), path, path.suffix.lower().lstrip("."))
        source_key = path.suffix.lower().lstrip(".")

    original_channels = point_cloud.shape[1]
    color_source = "existing_rgb" if original_channels >= 6 else "none"
    color_override = False
    if force_white_color:
        white = np.ones((point_cloud.shape[0], 3), dtype=np.float32)
        point_cloud = np.concatenate([point_cloud[:, :3], white], axis=1)
        color_source = "forced_white"
        color_override = original_channels >= 6

    metadata = {
        "num_points_before_processing": int(point_cloud.shape[0]),
        "num_channels_before_processing": int(point_cloud.shape[1]),
        "source_key": source_key,
        "color_source": color_source,
        "color_override": color_override,
    }
    return point_cloud, metadata


def configure_model_point_processing(model, args) -> None:
    if args.sample_points_num is not None:
        model.config.sample_points_num = args.sample_points_num
    elif not hasattr(model.config, "sample_points_num"):
        model.config.sample_points_num = 10000

    if args.with_color is not None:
        model.config.with_color = args.with_color
    elif not hasattr(model.config, "with_color"):
        model.config.with_color = True


def set_seed(seed: int) -> None:
    if seed < 0:
        return
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model):
    import torch

    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return next(model.parameters()).device


def build_prompt(question: str, model, conv_mode: str):
    from llava.constants import DEFAULT_POINT_TOKEN, DEFAULT_PT_END_TOKEN, DEFAULT_PT_START_TOKEN
    from llava.conversation import conv_templates

    if model.config.mm_use_pt_start_end:
        question = DEFAULT_PT_START_TOKEN + DEFAULT_POINT_TOKEN + DEFAULT_PT_END_TOKEN + "\n" + question
    else:
        question = DEFAULT_POINT_TOKEN + "\n" + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def generate_response(tokenizer, model, pts_tensor, prompt_text: str, args) -> str:
    import torch
    from llava.constants import POINT_TOKEN_INDEX
    from llava.conversation import SeparatorStyle
    from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_point_token

    model_device = get_model_device(model)
    conv, prompt = build_prompt(prompt_text, model, args.conv_mode)
    input_ids = tokenizer_point_token(
        prompt,
        tokenizer,
        POINT_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model_device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    do_sample = args.temperature > 0 and args.num_beams == 1
    generate_kwargs = {
        "points": pts_tensor,
        "do_sample": do_sample,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "stopping_criteria": [stopping_criteria],
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        if args.top_k is not None:
            generate_kwargs["top_k"] = args.top_k
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generate_kwargs)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
    output = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    output = output.strip()
    if output.endswith(stop_str):
        output = output[:-len(stop_str)]
    return output.strip()


def run_one_model(model_name: str, records: List[Dict[str, object]], args) -> None:
    import torch
    from llava.mm_utils import get_model_name_from_path, process_pts
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    model_path = existing_path_or_repo_relative(model_name)
    model_base = existing_path_or_repo_relative(args.model_base)
    model_tag = normalize_model_name(model_path)
    model_output_dir = args.output_root / model_tag
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false. Use `--device cpu` or fix CUDA.")

    set_seed(args.seed)
    disable_torch_init()
    os.chdir(REPO_ROOT)

    shape_model_name = get_model_name_from_path(model_path)
    print("=" * 80)
    print(f"[INFO] Loading ShapeLLM model {model_path}")
    tokenizer, model, _context_len = load_pretrained_model(
        model_path,
        model_base,
        shape_model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device_map,
        device=args.device,
    )
    model.eval()
    configure_model_point_processing(model, args)

    model_device = get_model_device(model)
    point_dtype = torch.float16 if model_device.type == "cuda" else torch.float32
    completed_records: List[Dict[str, object]] = []
    try:
        for index, record in enumerate(records, start=1):
            point_cloud, point_metadata = load_point_cloud_file(
                Path(record["point_cloud_path"]),
                force_white_color=args.force_white_color,
            )
            if getattr(model.config, "with_color", True) and point_cloud.shape[1] < 6:
                white = np.ones((point_cloud.shape[0], 3), dtype=np.float32)
                point_cloud = np.concatenate([point_cloud[:, :3], white], axis=1)
                point_metadata["color_source"] = "added_white_for_model"
                point_metadata["num_channels_before_processing"] = 6

            pts_tensor = process_pts(point_cloud.copy(), model.config).unsqueeze(0)
            pts_tensor = pts_tensor.to(model_device, dtype=point_dtype)

            response = generate_response(
                tokenizer=tokenizer,
                model=model,
                pts_tensor=pts_tensor,
                prompt_text=args.prompt,
                args=args,
            )

            result = OrderedDict(record)
            result["model_name"] = model_name
            result["model_path"] = model_path
            result["model_base"] = model_base
            result["point_cloud_num_points_before_processing"] = point_metadata["num_points_before_processing"]
            result["point_cloud_num_points_after_processing"] = int(pts_tensor.shape[1])
            result["point_cloud_source_key"] = point_metadata.get("source_key")
            result["point_cloud_color_source"] = point_metadata.get("color_source")
            result["point_cloud_color_override"] = point_metadata.get("color_override")
            result["shapellm_response"] = response
            completed_records.append(result)

            print("-" * 80)
            print(f"[{index}/{len(records)}] {model_tag} | {record['run_folder']} | {record['item_type']}")
            print(f"[GT] label={record['ground_truth_label']} part={record['expected_edited_part']}")
            print(f"[ANS] {response}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "model_base": model_base,
        "prompt": args.prompt,
        "note": (
            "By default all point clouds are forced to white RGB because BlendedPC does not provide real RGB. "
            "The prompt explicitly instructs ShapeLLM to ignore color and judge geometry only."
        ),
        "num_records": len(completed_records),
        "results": to_json_serializable(completed_records),
    }
    json_path = model_output_dir / "results.json"
    csv_path = model_output_dir / "results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    write_csv(completed_records, csv_path)
    print(f"[INFO] Wrote {json_path}")
    print(f"[INFO] Wrote {csv_path}")


def main():
    args = parse_args()
    args.model_name = [existing_path_or_repo_relative(model_name) for model_name in args.model_name]
    if args.model_base is not None:
        args.model_base = existing_path_or_repo_relative(args.model_base)

    records = collect_records(args)
    if not records:
        raise SystemExit("No BlendedPC records found.")

    print(f"[INFO] Collected {len(records)} point-cloud queries.")
    print(f"[INFO] Output root: {args.output_root}")
    for model_name in args.model_name:
        run_one_model(model_name, records, args)


if __name__ == "__main__":
    main()
