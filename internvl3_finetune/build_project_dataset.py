import json
import random
import sys
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference_common import project_videos_root, setup_run_logging


setup_run_logging("internvl3_build_dataset", __file__)


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    dataset: str
    label: str
    videos_root: Path


PROJECT_SPECS = {
    "ShapeNet": ProjectSpec(
        name="ShapeNet",
        dataset="ShapeNet",
        label="human-created",
        videos_root=project_videos_root("ULIP"),
    ),
    "InstantMesh": ProjectSpec(
        name="InstantMesh",
        dataset="InstantMesh",
        label="synthetic",
        videos_root=project_videos_root("InstantMesh"),
    ),
    "LGM": ProjectSpec(
        name="LGM",
        dataset="LGM",
        label="synthetic",
        videos_root=project_videos_root("LGM"),
    ),
    "SAM3D": ProjectSpec(
        name="SAM3D",
        dataset="SAM3D",
        label="synthetic",
        videos_root=project_videos_root("SAM3D"),
    ),
    "TRELLIS": ProjectSpec(
        name="TRELLIS",
        dataset="TRELLIS",
        label="synthetic",
        videos_root=project_videos_root("TRELLIS"),
    ),
    "TRELLIS_text": ProjectSpec(
        name="TRELLIS_text",
        dataset="TRELLIS_text",
        label="synthetic",
        videos_root=project_videos_root("TRELLIS_text"),
    ),
}


def parse_args():
    parser = ArgumentParser(description="Build a label-only InternVL3 training JSONL from project video roots.")
    parser.add_argument(
        "--output_jsonl",
        default=str(Path(__file__).resolve().parent / "all_projects_label_only.jsonl"),
        help="Output JSONL path for the combined dataset.",
    )
    parser.add_argument(
        "--projects",
        default=",".join(PROJECT_SPECS.keys()),
        help=f"Comma-separated subset of projects to include. Choices: {', '.join(PROJECT_SPECS)}",
    )
    parser.add_argument(
        "--max_per_project",
        type=int,
        default=0,
        help="If > 0, sample at most this many videos from each project.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed for per-project sampling and shuffling.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the final combined record order before writing.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.0,
        help="If > 0, also create object-level train/val splits with this validation fraction.",
    )
    parser.add_argument(
        "--train_jsonl",
        default="",
        help="Optional explicit train split output path. If omitted, derive from output_jsonl.",
    )
    parser.add_argument(
        "--val_jsonl",
        default="",
        help="Optional explicit validation split output path. If omitted, derive from output_jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def parse_project_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one project must be specified.")
    unknown = [name for name in names if name not in PROJECT_SPECS]
    if unknown:
        raise ValueError(f"Unknown projects: {unknown}. Choices: {sorted(PROJECT_SPECS)}")
    return names


def select_video_paths(spec: ProjectSpec, max_per_project: int, seed: int) -> list[Path]:
    if not spec.videos_root.exists():
        raise FileNotFoundError(f"Video root does not exist for {spec.name}: {spec.videos_root}")

    video_paths = sorted(spec.videos_root.glob("*.mp4"))
    if max_per_project > 0 and len(video_paths) > max_per_project:
        video_paths = sorted(random.Random(seed).sample(video_paths, max_per_project))
    return video_paths


def build_record(spec: ProjectSpec, video_path: Path) -> dict:
    object_id = video_path.stem
    return {
        "id": f"{spec.name}_{object_id}",
        "object_id": object_id,
        "dataset": spec.dataset,
        "project": spec.name,
        "video_path": str(video_path),
        "label": spec.label,
    }


def write_jsonl(path: Path, records: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def derive_split_paths(output_jsonl: Path, train_jsonl: str, val_jsonl: str) -> tuple[Path, Path]:
    if train_jsonl and val_jsonl:
        return Path(train_jsonl), Path(val_jsonl)
    if train_jsonl or val_jsonl:
        raise ValueError("Provide both --train_jsonl and --val_jsonl, or provide neither.")

    stem = output_jsonl.stem
    suffix = output_jsonl.suffix or ".jsonl"
    train_path = output_jsonl.with_name(f"{stem}_train{suffix}")
    val_path = output_jsonl.with_name(f"{stem}_val{suffix}")
    return train_path, val_path


def split_by_object(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["object_id"]].append(record)

    object_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(object_ids)

    val_object_count = max(1, int(round(len(object_ids) * val_fraction)))
    val_object_ids = set(object_ids[:val_object_count])

    train_records = []
    val_records = []
    for object_id in object_ids:
        target = val_records if object_id in val_object_ids else train_records
        target.extend(grouped[object_id])

    return train_records, val_records


def print_summary(records: list[dict], prefix: str) -> None:
    label_counts = Counter(record["label"] for record in records)
    project_counts = Counter(record["project"] for record in records)
    print(f"{prefix} total_records={len(records)}")
    print(f"{prefix} label_counts={dict(label_counts)}")
    print(f"{prefix} project_counts={dict(project_counts)}")


def main() -> None:
    args = parse_args()

    if args.max_per_project < 0:
        raise ValueError("--max_per_project must be >= 0.")
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val_fraction must be in [0, 1).")

    selected_projects = parse_project_names(args.projects)
    output_jsonl = Path(args.output_jsonl)

    all_records = []
    for project_index, project_name in enumerate(selected_projects):
        spec = PROJECT_SPECS[project_name]
        video_paths = select_video_paths(
            spec=spec,
            max_per_project=args.max_per_project,
            seed=args.sample_seed + project_index,
        )
        records = [build_record(spec, video_path) for video_path in video_paths]
        all_records.extend(records)
        print(
            f"Collected {len(records)} records for {project_name} "
            f"from {spec.videos_root} with label={spec.label}."
        )

    if args.shuffle:
        random.Random(args.sample_seed).shuffle(all_records)

    print_summary(all_records, prefix="combined")
    write_jsonl(output_jsonl, all_records, overwrite=args.overwrite)
    print(f"Wrote combined dataset to {output_jsonl}")

    if args.val_fraction > 0.0:
        train_records, val_records = split_by_object(
            records=all_records,
            val_fraction=args.val_fraction,
            seed=args.sample_seed,
        )
        train_path, val_path = derive_split_paths(output_jsonl, args.train_jsonl, args.val_jsonl)
        write_jsonl(train_path, train_records, overwrite=args.overwrite)
        write_jsonl(val_path, val_records, overwrite=args.overwrite)
        print_summary(train_records, prefix="train")
        print_summary(val_records, prefix="val")
        print(f"Wrote train split to {train_path}")
        print(f"Wrote val split to {val_path}")


if __name__ == "__main__":
    main()
