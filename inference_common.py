import atexit
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = REPO_DIR.parent
ROOT_LOG_DIR = Path(os.environ.get("INFERENCE_LOG_ROOT_BASE", str(DEFAULT_PROJECT_ROOT / "logs")))
PROJECT_ROOT = Path(os.environ.get("SYN3D_PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT)))
PROJECT_NAMES = ("ULIP", "InstantMesh", "LGM", "SAM3D", "TRELLIS", "TRELLIS_text")
PROJECT_ORDER = {
    "ULIP": 1,
    "InstantMesh": 2,
    "LGM": 3,
    "SAM3D": 4,
    "TRELLIS": 5,
    "TRELLIS_text": 6,
}
PROJECT_VIDEO_RELATIVE_CANDIDATES = {
    "ULIP": ("ulip/ULIP_Shapenet_Triplets/videos_white",),
    "InstantMesh": ("outputs/videos_white", "outputs/videos_black"),
    "LGM": ("outputs/videos_white", "outputs/videos_black"),
    "SAM3D": ("outputs/videos_white", "outputs/videos_black"),
    "TRELLIS": ("outputs/videos_white", "outputs/videos_black"),
    "TRELLIS_text": ("outputs_text/videos_white", "outputs_text/videos_black"),
}


def project_root(project_name: str) -> Path:
    return PROJECT_ROOT / project_name


def project_videos_root(project_name: str) -> Path:
    root = project_root(project_name)
    candidates = PROJECT_VIDEO_RELATIVE_CANDIDATES.get(
        project_name,
        ("ulip/ULIP_Shapenet_Triplets/videos_white",),
    )
    for relative_path in candidates:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return root / candidates[0]


def iter_project_configs(output_filename: str):
    for project_name in PROJECT_NAMES:
        root = project_root(project_name)
        yield project_name, project_videos_root(project_name), root / output_filename


class TeeStream:
    def __init__(self, primary, secondary, lock: Lock):
        self.primary = primary
        self.secondary = secondary
        self.lock = lock
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data):
        if not data:
            return 0

        with self.lock:
            self.primary.write(data)
            self.primary.flush()
            self.secondary.write(data)
            self.secondary.flush()
        return len(data)

    def flush(self):
        with self.lock:
            self.primary.flush()
            self.secondary.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self):
        return self.primary.fileno()


def _derive_label(script_path: str | Path) -> str:
    project_name = Path(script_path).resolve().parent.name
    project_index = PROJECT_ORDER.get(project_name)
    if project_index is None:
        return project_name
    return f"{project_index:02d}_{project_name}"


def setup_run_logging(model_family: str, script_path: str | Path) -> Path | None:
    if os.environ.get("INFERENCE_DISABLE_INTERNAL_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None

    existing_path = getattr(sys, "_inference_log_path", None)
    if existing_path is not None:
        return Path(existing_path)

    log_root = Path(os.environ.get("INFERENCE_LOG_ROOT", str(ROOT_LOG_DIR / model_family)))
    run_id = os.environ.get("INFERENCE_LOG_RUN_ID")
    if not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    log_label = os.environ.get("INFERENCE_LOG_LABEL", _derive_label(script_path))
    log_path = Path(
        os.environ.get("INFERENCE_LOG_PATH", str(log_root / run_id / f"{log_label}.log"))
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    lock = Lock()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, file_handle, lock)
    sys.stderr = TeeStream(original_stderr, file_handle, lock)
    sys._inference_log_path = str(log_path)

    @atexit.register
    def _close_log_file():
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            file_handle.flush()
        finally:
            file_handle.close()

    print(f"Writing logs to {log_path}")
    return log_path
