import json
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(
        description="Write an object-id exclusion list from one or more training JSONL files."
    )
    parser.add_argument(
        "input_jsonl",
        nargs="+",
        help="Training JSONL files containing object_id, video_path, or frame_paths.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "train_exclude_object_ids.txt"),
        help="Output text file with one object id per line.",
    )
    return parser.parse_args()


def iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def object_id_from_record(record: dict) -> str | None:
    object_id = str(record.get("object_id", "")).strip()
    if object_id:
        return object_id

    video_path = str(record.get("video_path", "")).strip()
    if video_path:
        return Path(video_path).stem

    frame_paths = record.get("frame_paths")
    if isinstance(frame_paths, list) and frame_paths:
        first_frame_path = str(frame_paths[0]).strip()
        if first_frame_path:
            return Path(first_frame_path).parent.name

    return None


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.input_jsonl]
    output_path = Path(args.output)

    object_ids = set()
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        for record in iter_jsonl_records(input_path):
            object_id = object_id_from_record(record)
            if object_id:
                object_ids.add(object_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for object_id in sorted(object_ids):
            handle.write(object_id + "\n")

    print(f"Wrote {len(object_ids)} excluded object ids to {output_path}")


if __name__ == "__main__":
    main()

