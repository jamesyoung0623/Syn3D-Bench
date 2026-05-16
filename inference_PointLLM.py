import argparse
import json
import os
import random
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


SYN3D_ROOT = Path(__file__).resolve().parent
POINTLLM_ROOT = Path(os.environ.get("POINTLLM_ROOT", "/home/jamesyoung0623/PointLLM")).resolve()
if str(POINTLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTLLM_ROOT))


HOME = Path("/home/jamesyoung0623")
SUPPORTED_POINT_SUFFIXES = (".npy", ".npz", ".ply", ".pt", ".pth")
PROJECT_NAMES = ("Shapenet", "InstantMesh", "LGM", "SAM3D", "TRELLIS", "TRELLIS_text")
PROJECT_ORDER = {
    "Shapenet": 1,
    "InstantMesh": 2,
    "LGM": 3,
    "SAM3D": 4,
    "TRELLIS": 5,
    "TRELLIS_text": 6,
}

DEFAULT_PROJECT_PC_CANDIDATES = {
    "Shapenet": (
        HOME / "ULIP/ulip/ULIP_Shapenet_Triplets/shapenet_pc",
        HOME / "BlendedPC/inputs/ulip_shapenet",
    ),
    "InstantMesh": (
        HOME / "InstantMesh/outputs/PCs",
        HOME / "datasets/raw_point_clouds/InstantMesh",
    ),
    "LGM": (
        HOME / "LGM/outputs/PCs",
        HOME / "datasets/raw_point_clouds/LGM",
    ),
    "SAM3D": (
        HOME / "SAM3D/outputs/PCs",
        HOME / "datasets/raw_point_clouds/SAM3D",
    ),
    "TRELLIS": (
        HOME / "TRELLIS/outputs/PCs",
        HOME / "datasets/raw_point_clouds/TRELLIS",
    ),
    "TRELLIS_text": (
        HOME / "TRELLIS_text/outputs_text/PCs",
        HOME / "TRELLIS_text/outputs/PCs",
        HOME / "datasets/raw_point_clouds/TRELLIS_text",
    ),
}

DEFAULT_PROJECT_OUTPUT_ROOTS = {
    "Shapenet": HOME / "ULIP",
    "InstantMesh": HOME / "InstantMesh",
    "LGM": HOME / "LGM",
    "SAM3D": HOME / "SAM3D",
    "TRELLIS": HOME / "TRELLIS",
    "TRELLIS_text": HOME / "TRELLIS_text",
}


DEFAULT_PROMPT = """
You are analyzing a 3D point cloud of one object.

Your goal is to infer the origin of the underlying 3D model, not to describe the point-cloud file format.

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
  The geometric evidence is insufficient to reliably determine whether the asset was primarily human-authored or machine-generated.

Important rules:
1. Base your decision only on diagnostic evidence about the 3D model's origin.
2. Do NOT use the same kind of reason to justify opposite labels.
3. Focus on geometric features of the underlying asset such as topology plausibility, part coherence, symmetry vs over-regularization, repeated geometry, implausible structure, missing functional details, semantic incoherence, signs of manual design intent, or signs of procedural/generative artifacts.
4. Ignore color. If color is present, it may be synthetic white filler for model compatibility.
5. For "real", the reason should point to evidence of deliberate manual design, functional structure, meaningful detail placement, or coherent asset construction.
6. For "synthetic", the reason should point to evidence of generative artifacts, implausible geometry, repeated or nonsensical structure, over-smoothing, inconsistent semantics, or missing/merged functional parts.
7. For "uncertain", the reason should explain exactly why the point cloud does not provide diagnostic evidence.

Output requirements:
- Return valid JSON only.
- Provide exactly one reason in one sentence.
- The reason must cite a diagnostic geometric cue.
- Do not repeat the label wording in the reason.

Return JSON with this schema:
{
  "label": "real" | "synthetic" | "uncertain",
  "reason": "one-sentence reason"
}
""".strip()


def detect_default_device() -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def parse_key_value_overrides(items: Optional[List[str]]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=PATH override, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty project key in override: {item}")
        overrides[key] = Path(os.path.expanduser(value.strip())).resolve()
    return overrides


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PointLLM on Syn3D-Bench raw point clouds one object at a time. "
            "The loop and output records mirror the Syn3D-Bench image/video inference scripts."
        )
    )
    parser.add_argument(
        "--model_name",
        nargs="+",
        default=[str(POINTLLM_ROOT / "checkpoints/PointLLM_7B_v1.2")],
        help="One or more PointLLM model ids or local checkpoint directories.",
    )
    parser.add_argument("--projects", nargs="+", default=list(PROJECT_NAMES), choices=list(PROJECT_NAMES))
    parser.add_argument(
        "--project_pc_root",
        action="append",
        default=None,
        help=(
            "Override a project point-cloud root as PROJECT=/path. Repeatable. "
            "Useful for SAM3D/TRELLIS if raw point clouds are stored outside the default outputs/PCs path."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Optional central output root. If omitted, results are written to project roots, "
            "matching the Syn3D-Bench inference scripts."
        ),
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional output filename. Default: pointllm_<model_tag>_all_results.json.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search point-cloud roots recursively.")
    parser.add_argument("--max_items", type=int, default=1000)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--device", type=str, default=detect_default_device())
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--force_white_color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force/add white RGB channels because PointLLM expects colored input. Default: enabled.",
    )
    parser.add_argument(
        "--skip_failed_projects",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue without raising if a project root is missing or has no usable point clouds.",
    )
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_mapping[args.torch_dtype]
    args.device = torch.device(args.device)
    if args.device.type == "cpu" and args.torch_dtype != torch.float32:
        args.torch_dtype = torch.float32
    args.project_pc_root = parse_key_value_overrides(args.project_pc_root)
    return args


def normalize_model_name(model_name: str) -> str:
    return Path(model_name.rstrip("/")).name.replace(":", "_")


def expected_label_for_project(project_name: str) -> str:
    return "real" if project_name == "Shapenet" else "synthetic"


def resolve_project_pc_root(project_name: str, overrides: Dict[str, Path]) -> Path:
    if project_name in overrides:
        return overrides[project_name]

    candidates = DEFAULT_PROJECT_PC_CANDIDATES[project_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_project_output_path(project_name: str, model_tag: str, args) -> Path:
    output_filename = args.output_filename or f"pointllm_{model_tag}_all_results.json"
    if args.output_root is not None:
        return args.output_root.resolve() / model_tag / project_name / output_filename

    output_root = DEFAULT_PROJECT_OUTPUT_ROOTS.get(project_name, HOME / project_name)
    return output_root / output_filename


def iter_point_cloud_paths(root: Path, recursive: bool) -> Iterable[Path]:
    patterns = [f"**/*{suffix}" if recursive else f"*{suffix}" for suffix in SUPPORTED_POINT_SUFFIXES]
    seen = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def select_point_cloud_paths(root: Path, max_items: int, sample_seed: int, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Point-cloud root does not exist: {root}")

    paths = sorted(iter_point_cloud_paths(root, recursive))
    if not paths:
        raise FileNotFoundError(f"No point cloud files found under: {root}")

    if max_items > 0 and len(paths) > max_items:
        paths = sorted(random.Random(sample_seed).sample(paths, max_items))
    return paths


def object_id_from_path(path: Path) -> str:
    stem = path.stem
    return re.sub(r"_[0-9]+$", "", stem)


def _uniform_rgb(num_points: int, value: float = 1.0) -> np.ndarray:
    return np.full((num_points, 3), value, dtype=np.float32)


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    max_value = float(np.max(rgb))
    if max_value <= 1.0:
        return rgb
    if max_value <= 255.0:
        return rgb / 255.0
    raise ValueError(f"Unsupported RGB range with max value {max_value}.")


def _coerce_array(value, path: Path, source_key: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype == object and value.shape == ():
        value = np.asarray(value.item())
    if value.ndim == 3:
        value = value[0]
    if value.ndim != 2:
        if value.ndim == 1:
            raise ValueError(
                f"{path}:{source_key} is a 1D tensor with shape {value.shape}; "
                "this looks like a feature embedding, not a raw point cloud."
            )
        raise ValueError(f"{path}:{source_key} is not a 2D point array; shape={value.shape}")
    if value.shape[0] in {3, 6} and value.shape[1] not in {3, 6}:
        value = value.T
    if value.shape[1] < 3:
        raise ValueError(f"{path}:{source_key} has fewer than 3 columns; shape={value.shape}")
    return value.astype(np.float32, copy=False)


def _array_from_mapping(mapping: Dict, path: Path) -> Tuple[np.ndarray, str]:
    preferred_keys = (
        "points",
        "pointcloud",
        "point_cloud",
        "coords",
        "xyz",
        "pred",
        "arr_0",
        "vertices",
    )
    keys = list(mapping.keys())
    for key in preferred_keys + tuple(k for k in keys if k not in preferred_keys):
        if key not in mapping:
            continue
        try:
            return _coerce_array(mapping[key], path, str(key)), str(key)
        except Exception:
            continue
    raise ValueError(f"Could not find a raw point array in {path}; available keys={keys}")


def _load_npz(path: Path) -> Tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=True) as data:
        return _array_from_mapping({key: data[key] for key in data.files}, path)


def _load_torch(path: Path) -> Tuple[np.ndarray, str]:
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        return _array_from_mapping(data, path)
    return _coerce_array(data, path, "tensor"), "tensor"


def _load_ply(path: Path) -> Tuple[np.ndarray, str]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError(f"Could not read point vertices from {path}")
    if rgb.size > 0 and rgb.shape[0] == xyz.shape[0]:
        return np.concatenate([xyz, _normalize_rgb(rgb)], axis=1).astype(np.float32), "ply_xyzrgb"
    return xyz, "ply_xyz"


def load_point_cloud_file(path: Path, pointnum: int, force_white_color: bool) -> Tuple[np.ndarray, Dict[str, object]]:
    from pointllm.data import farthest_point_sample, pc_norm

    path = path.resolve()
    ext = path.suffix.lower()
    metadata: Dict[str, object] = {"path": str(path), "format": ext}

    if ext == ".npy":
        point_array = _coerce_array(np.load(path, allow_pickle=True), path, "npy")
        source_key = "npy"
    elif ext == ".npz":
        point_array, source_key = _load_npz(path)
    elif ext in {".pt", ".pth"}:
        point_array, source_key = _load_torch(path)
    elif ext == ".ply":
        point_array, source_key = _load_ply(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    if point_array.shape[0] == 0:
        raise ValueError("Point cloud is empty.")

    xyz = np.asarray(point_array[:, :3], dtype=np.float32)
    if point_array.shape[1] >= 6 and not force_white_color:
        rgb = _normalize_rgb(point_array[:, 3:6])
        color_source = "file"
        color_override = False
    else:
        rgb = _uniform_rgb(xyz.shape[0], 1.0)
        color_source = "forced_white" if force_white_color else "missing->white"
        color_override = bool(point_array.shape[1] >= 6 and force_white_color)

    points = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    metadata["source_key"] = source_key
    metadata["num_points_before_sampling"] = int(points.shape[0])
    metadata["num_channels_before_sampling"] = int(points.shape[1])
    metadata["color_source"] = color_source
    metadata["color_override"] = color_override

    if points.shape[0] > pointnum:
        points = farthest_point_sample(points, pointnum)
        metadata["sampling"] = f"farthest_point_sample->{pointnum}"
    else:
        metadata["sampling"] = "none"

    points = pc_norm(points)
    metadata["num_points_after_sampling"] = int(points.shape[0])
    return points, metadata


def extract_first_json_object(text: str):
    decoder = json.JSONDecoder()
    text = text.strip()

    try:
        return json.loads(text), None
    except Exception:
        pass

    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[idx:])
            return parsed, None
        except Exception:
            continue
    return None, "No valid JSON object found"


def validate_parsed_output(parsed) -> Optional[str]:
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        return "Parsed JSON is not an object"
    label = parsed.get("label")
    if label not in {"real", "synthetic", "uncertain"}:
        return f"Invalid label: {label!r}"
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "Missing non-empty reason"
    return None


def run_inference_on_point_cloud(model_bundle, point_cloud_path: Path, project_name: str, args) -> Dict[str, object]:
    from pointllm.eval.query_custom_pointcloud import generate_response

    model, tokenizer, conv, keywords, point_backbone_config, mm_use_point_start_end = model_bundle
    point_cloud, metadata = load_point_cloud_file(
        point_cloud_path,
        args.pointnum,
        force_white_color=args.force_white_color,
    )

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        conv_template=conv,
        keywords=keywords,
        point_backbone_config=point_backbone_config,
        mm_use_point_start_end=mm_use_point_start_end,
        point_cloud=point_cloud,
        question=args.prompt,
        device=args.device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length,
    )

    parsed, parse_error = extract_first_json_object(response)
    validation_error = validate_parsed_output(parsed)
    if parse_error is None and validation_error is not None:
        parse_error = validation_error

    object_id = object_id_from_path(point_cloud_path)
    return {
        "project_name": project_name,
        "point_cloud_path": str(point_cloud_path),
        "object_id": object_id,
        "point_cloud_name": point_cloud_path.name,
        "expected_label": expected_label_for_project(project_name),
        "point_metadata": metadata,
        "raw_output": response,
        "parsed_output": parsed,
        "parse_error": parse_error,
    }


def load_model_bundle(model_name: str, args):
    from pointllm.eval.query_custom_pointcloud import init_model

    model_args = argparse.Namespace(
        model_name=os.path.expanduser(model_name),
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    if model_args.device.type == "cuda":
        try:
            torch.zeros(1, device=model_args.device)
        except Exception as exc:
            raise SystemExit(
                "CUDA is not usable in the current environment. "
                "Use `--device cpu` for debugging or fix the PyTorch/CUDA driver setup."
            ) from exc

    print("=" * 80)
    print(f"[INFO] Loading PointLLM model {model_name}")
    return init_model(model_args)


def run_current_project(project_name: str, model_bundle, model_name: str, args) -> None:
    pc_root = resolve_project_pc_root(project_name, args.project_pc_root)
    model_tag = normalize_model_name(model_name)
    output_path = resolve_project_output_path(project_name, model_tag, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_cloud_paths = select_point_cloud_paths(
        root=pc_root,
        max_items=args.max_items,
        sample_seed=args.sample_seed,
        recursive=args.recursive,
    )
    print(f"Selected {len(point_cloud_paths)} point clouds from {pc_root}.")

    all_results: List[Dict[str, object]] = []
    for index, point_cloud_path in enumerate(point_cloud_paths, start=1):
        print(f"\n[{index}/{len(point_cloud_paths)}] Processing: {point_cloud_path}")
        try:
            result = run_inference_on_point_cloud(
                model_bundle=model_bundle,
                point_cloud_path=point_cloud_path,
                project_name=project_name,
                args=args,
            )
            all_results.append(result)

            print("Raw output:")
            print(result["raw_output"])
            if result["parsed_output"] is not None and result["parse_error"] is None:
                print("Parsed JSON:")
                print(result["parsed_output"])
            else:
                print(f"Model output was not strict target JSON. Error: {result['parse_error']}")
        except Exception as exc:
            error_result = {
                "project_name": project_name,
                "point_cloud_path": str(point_cloud_path),
                "object_id": object_id_from_path(point_cloud_path),
                "point_cloud_name": point_cloud_path.name,
                "expected_label": expected_label_for_project(project_name),
                "error": str(exc),
            }
            all_results.append(error_result)
            print(f"Failed: {exc}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. All results saved to: {output_path}")


def main():
    args = parse_args()
    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed)
    torch.manual_seed(args.sample_seed)

    failures: List[Tuple[str, Exception]] = []
    for model_name in args.model_name:
        model_bundle = load_model_bundle(model_name, args)
        try:
            for project_name in args.projects:
                print(f"\n=== PointLLM inference: {project_name} ===")
                try:
                    run_current_project(project_name, model_bundle, model_name, args)
                except Exception as exc:
                    failures.append((project_name, exc))
                    print(f"Failed {project_name}: {exc}")
        finally:
            model, tokenizer, *_rest = model_bundle
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failures and not args.skip_failed_projects:
        failed = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"PointLLM inference failed for: {failed}")


if __name__ == "__main__":
    main()
