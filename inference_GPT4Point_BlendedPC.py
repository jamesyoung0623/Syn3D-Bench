import argparse
import csv
import importlib.util
import json
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import torch


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
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def parse_args():
    repo_root = Path(__file__).resolve().parent
    default_blendedpc_root = Path("/home/jamesyoung0623/BlendedPC")
    default_output_root = repo_root / "evaluation" / "blendedpc_edit_detection_gpt4point"
    default_gpt4point_root = Path("/home/jamesyoung0623/GPT4Point")
    default_pointllm_root = repo_root.parent / "PointLLM"

    parser = argparse.ArgumentParser(
        description=(
            "Run GPT4Point on BlendedPC originals and edited outputs one point cloud at a time, "
            "asking whether each point cloud appears edited and which part seems edited."
        )
    )
    parser.add_argument(
        "--model_name",
        nargs="+",
        required=True,
        help=(
            "One or more GPT4Point checkpoint specs. Each item may be a stage-2 checkpoint `.pth` file, "
            "a directory containing it, or `alexzyqi/GPT4Point` to use the default local checkpoint."
        ),
    )
    parser.add_argument("--gpt4point_root", type=Path, default=default_gpt4point_root)
    parser.add_argument(
        "--pointllm_root",
        type=Path,
        default=default_pointllm_root,
        help=(
            "Path to the PointLLM repo containing `pointllm.eval.query_custom_pointcloud_gpt4point`. "
            "Defaults to the sibling PointLLM checkout next to GPT4Point."
        ),
    )
    parser.add_argument(
        "--point_encoder_checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional override for `point_encoder_pointbert_wcolor.pth`. "
            "Defaults to `<gpt4point_root>/checkpoints/point_encoder_pointbert_wcolor.pth`."
        ),
    )
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
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--output_root", type=Path, default=default_output_root)
    parser.add_argument("--limit_runs", type=int, default=0, help="Optional limit on the number of BlendedPC run folders to evaluate.")
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
        gpt4point_prompt=prompt_text,
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


def resolve_gpt4point_checkpoint(model_name: str, gpt4point_root: Path, stage2_filename: str) -> Path:
    expanded_model_name = os.path.expanduser(model_name)
    candidate_path = Path(expanded_model_name)

    if candidate_path.is_file():
        return candidate_path.resolve()

    if candidate_path.is_dir():
        dir_candidates = [
            candidate_path / stage2_filename,
            candidate_path / "checkpoints" / stage2_filename,
        ]
        for item in dir_candidates:
            if item.is_file():
                return item.resolve()
        raise FileNotFoundError(
            f"Could not find `{stage2_filename}` inside directory `{candidate_path}`."
        )

    if model_name in {"alexzyqi/GPT4Point", "GPT4Point", "gpt4point"}:
        default_checkpoint = gpt4point_root / "checkpoints" / stage2_filename
        if default_checkpoint.is_file():
            return default_checkpoint.resolve()
        raise FileNotFoundError(
            f"Default GPT4Point checkpoint not found at `{default_checkpoint}`."
        )

    raise FileNotFoundError(
        "Unsupported GPT4Point model spec. Provide a checkpoint `.pth` file, a directory containing it, "
        "or `alexzyqi/GPT4Point`."
    )


def load_gpt4point_helper(pointllm_root: Path):
    helper_path = pointllm_root.resolve() / "pointllm" / "eval" / "query_custom_pointcloud_gpt4point.py"
    if not helper_path.is_file():
        raise FileNotFoundError(f"GPT4Point helper not found: {helper_path}")

    module_name = "_pointllm_query_custom_pointcloud_gpt4point"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GPT4Point helper from {helper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_one_model(model_name: str, records: List[Dict[str, object]], args) -> None:
    helper = load_gpt4point_helper(args.pointllm_root)
    EXPECTED_POINT_ENCODER_FILENAME = helper.EXPECTED_POINT_ENCODER_FILENAME
    EXPECTED_STAGE2_FILENAME = helper.EXPECTED_STAGE2_FILENAME
    generate_response = helper.generate_response
    init_model = helper.init_model
    load_point_cloud_file = helper.load_point_cloud_file

    model_tag = normalize_model_name(model_name)
    model_output_dir = args.output_root / model_tag
    model_output_dir.mkdir(parents=True, exist_ok=True)

    point_encoder_checkpoint = args.point_encoder_checkpoint
    if point_encoder_checkpoint is None:
        point_encoder_checkpoint = args.gpt4point_root / "checkpoints" / EXPECTED_POINT_ENCODER_FILENAME
    point_encoder_checkpoint = point_encoder_checkpoint.resolve()

    checkpoint_path = resolve_gpt4point_checkpoint(
        model_name=model_name,
        gpt4point_root=args.gpt4point_root.resolve(),
        stage2_filename=EXPECTED_STAGE2_FILENAME,
    )

    model_args = argparse.Namespace(
        gpt4point_root=args.gpt4point_root.resolve(),
        checkpoint=checkpoint_path,
        point_encoder_checkpoint=point_encoder_checkpoint,
        device=torch.device(args.device),
        pointnum=args.pointnum,
    )
    if model_args.device.type == "cuda":
        try:
            torch.zeros(1, device=model_args.device)
        except Exception as exc:
            raise SystemExit(
                "CUDA is not usable in the current environment. "
                "Your NVIDIA driver is likely too old for the installed PyTorch/CUDA build. "
                "Update the GPU driver or install a PyTorch build compatible with this driver. "
                "You can also try `--device cpu` for debugging, but full GPT4Point inference on CPU is usually impractical."
            ) from exc

    print("=" * 80)
    print(f"[INFO] Loading GPT4Point checkpoint {checkpoint_path}")
    model, pts_processor = init_model(model_args)

    completed_records: List[Dict[str, object]] = []
    try:
        for index, record in enumerate(records, start=1):
            point_cloud, point_metadata = load_point_cloud_file(
                record["point_cloud_path"],
                args.pointnum,
                force_white_color=True,
            )
            responses = generate_response(
                model=model,
                pts_processor=pts_processor,
                point_cloud=point_cloud,
                prompt=args.prompt,
                device=model_args.device,
                use_nucleus_sampling=False,
                top_p=1.0,
                temperature=1.0,
                num_beams=5,
                max_length=args.max_length,
                min_length=1,
                num_captions=1,
            )
            response = responses[0] if responses else ""

            result = OrderedDict(record)
            result["model_name"] = model_name
            result["checkpoint_path"] = str(checkpoint_path)
            result["point_cloud_num_points"] = point_metadata["num_points_after_sampling"]
            result["point_cloud_color_source"] = point_metadata.get("color_source")
            result["point_cloud_color_override"] = point_metadata.get("color_override")
            result["point_cloud_sampling"] = point_metadata.get("sampling")
            result["gpt4point_response"] = response
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
        "checkpoint_path": str(checkpoint_path),
        "prompt": args.prompt,
        "note": (
            "All point clouds were forced to white RGB because GPT4Point was trained with colored input. "
            "The prompt explicitly instructs the model to ignore color and judge geometry only."
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
    records = collect_records(args)
    if not records:
        raise SystemExit("No BlendedPC records found.")

    print(f"[INFO] Collected {len(records)} point-cloud queries.")
    print(f"[INFO] Output root: {args.output_root}")
    for model_name in args.model_name:
        run_one_model(model_name, records, args)


if __name__ == "__main__":
    main()
