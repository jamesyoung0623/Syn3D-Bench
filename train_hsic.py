from copy import deepcopy
import os
import glob
import csv
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, accuracy_score

from networks.trainer import Trainer
from options.train_options import TrainOptions
from networks.base_model import Model

TEN_TEST_COMBOS = [
    ("Objaverse", "InstantMesh"),
    ("Objaverse", "LGM"),
    ("Objaverse", "SAM3D"),
    ("Objaverse", "TRELLIS"),
    ("Objaverse", "TRELLIS_text"),
    ("Shapenet", "InstantMesh"),
    ("Shapenet", "LGM"),
    ("Shapenet", "SAM3D"),
    ("Shapenet", "TRELLIS"),
    ("Shapenet", "TRELLIS_text"),
]
PAIRED_TEST_COMBOS = {
    ("Shapenet", "InstantMesh"),
    ("Shapenet", "LGM"),
    ("Shapenet", "SAM3D"),
    ("Shapenet", "TRELLIS"),
}
AUTO_UNPAIRED_REAL_DATASETS = {"objaverse"}
AUTO_UNPAIRED_FAKE_DATASETS = {"trellis_text"}
DATASET_ABBREVIATIONS = {
    "objaverse": "O",
    "instantmesh": "I",
    "lgm": "L",
    "sam3d": "S",
    "trellis": "T",
    "trellis_text": "TT",
    "shapenet": "S",
}


# ------------- helpers -------------
def _sanitize_result_tag(tag: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(tag))


def _parse_int_list(csv_str: str):
    return [int(x.strip()) for x in csv_str.split(',') if x.strip()]


def _parse_csv_list(value: str):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_dataset_paths(items, pc_root, pc_variant, pc_backbone):
    paths = []
    for item in items:
        if "/" in item:
            paths.append(item)
        else:
            paths.append(os.path.join(pc_root, item, pc_variant, pc_backbone))
    return paths


def _validate_variant_backbone(paths, pc_variant, pc_backbone):
    for p in paths:
        if "/" not in p:
            continue
        parts = os.path.normpath(p).split(os.sep)
        if len(parts) < 2:
            continue
        variant = parts[-2]
        backbone = parts[-1]
        if variant != pc_variant or backbone != pc_backbone:
            raise ValueError(
                f"Path {p} does not match --pc_variant {pc_variant} and --pc_backbone {pc_backbone}"
            )


def _dataset_name_from_path(path: str):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return os.path.basename(os.path.normpath(path))


def _normalize_dataset_token(name: str):
    return str(name).strip().lower().replace("-", "_")


def _dataset_abbreviation(name: str):
    token = _normalize_dataset_token(name)
    mapped = DATASET_ABBREVIATIONS.get(token)
    if mapped:
        return mapped
    parts = [p for p in token.split("_") if p]
    if not parts:
        return "X"
    if len(parts) == 1:
        return parts[0][0].upper()
    return "".join(p[0].upper() for p in parts)


def _dataset_group_abbreviation(names):
    return "".join(_dataset_abbreviation(n) for n in names)


def _should_auto_unpaired(train_real_names, train_fake_names):
    real_norm = {_normalize_dataset_token(n) for n in train_real_names}
    fake_norm = {_normalize_dataset_token(n) for n in train_fake_names}
    return bool(real_norm & AUTO_UNPAIRED_REAL_DATASETS) or bool(fake_norm & AUTO_UNPAIRED_FAKE_DATASETS)


def _load_ids(path: str):
    ids = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid id line (expected cat<TAB>idx): {line}")
            ids.add((parts[0], parts[1]))
    if not ids:
        raise ValueError(f"No ids loaded from {path}")
    return ids


def _assert_split_ids_disjoint(split_ids):
    train_ids = split_ids["train"]
    val_ids = split_ids["val"]
    test_ids = split_ids["test"]

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "train/val/test IDs must be disjoint, but overlaps were found: "
            f"train∩val={len(overlap_tv)} train∩test={len(overlap_tt)} val∩test={len(overlap_vt)}"
        )


class FeatureListDataset(Dataset):
    """Dataset backed by a list of (path, label) tuples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        feat = _load_feature_tensor(path)
        target = torch.tensor(label, dtype=torch.float32)
        return feat, target


def _load_feature_tensor(path: str):
    feat = torch.load(path, map_location="cpu")
    if not torch.is_tensor(feat):
        feat = torch.tensor(feat)
    feat = feat.float()
    if feat.dim() == 2 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    elif feat.dim() > 1:
        feat = feat.reshape(-1)
    return feat


def _parse_id(fname: str):
    """Return (category, index) from a filename like '02691156-1a04e3...pt'."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if '-' not in stem:
        raise ValueError(f"Filename does not contain category/index separator '-': {fname}")
    cat, idx = stem.split('-', 1)
    return cat, idx


def _safe_parse_id(fname: str):
    """Return (category, index) or None if the filename doesn't match cat-idx format."""
    try:
        return _parse_id(fname)
    except ValueError:
        return None


def _stem_id(fname: str):
    return os.path.splitext(os.path.basename(fname))[0]


def _split_ids(ids):
    """Split list of ids into 80/10/10 (deterministic shuffle)."""
    rng = random.Random(99)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    # ensure val/test not empty when we have at least 3 items
    if n >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(0, n_train - 1)
        if n_test == 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def _collect_samples(real_dirs, fake_dirs, allowed_ids):
    if not real_dirs or not fake_dirs:
        raise ValueError("real_dirs and fake_dirs must be non-empty lists.")

    real_files = []
    fake_files = []
    for d in real_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Real dataset directory not found: {d}")
        real_files.extend(glob.glob(os.path.join(d, "*.pt")))
    for d in fake_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Fake dataset directory not found: {d}")
        fake_files.extend(glob.glob(os.path.join(d, "*.pt")))

    if not real_files or not fake_files:
        raise FileNotFoundError("No .pt feature files found in real or fake dataset directories.")

    # Standard path: strict cat-idx pairing. Fallback path: idx-only pairing for
    # mixed naming formats (e.g., UUID-only filenames on one side).
    real_parsed = [_safe_parse_id(fp) for fp in real_files]
    fake_parsed = [_safe_parse_id(fp) for fp in fake_files]
    has_unparseable = any(p is None for p in real_parsed) or any(p is None for p in fake_parsed)

    if not has_unparseable:
        cat_to_real = {}
        for fp, parsed in zip(real_files, real_parsed):
            cat, idx = parsed
            cat_to_real.setdefault(cat, {})[idx] = fp

        cat_to_fake = {}
        for fp, parsed in zip(fake_files, fake_parsed):
            cat, idx = parsed
            cat_to_fake.setdefault(cat, {})[idx] = fp

        samples = []
        for cat, real_map in cat_to_real.items():
            fake_map = cat_to_fake.get(cat, {})
            common_ids = sorted(set(real_map.keys()) & set(fake_map.keys()))
            if not common_ids:
                continue
            for idx in common_ids:
                if (cat, idx) not in allowed_ids:
                    continue
                samples.append((real_map[idx], 0))
                samples.append((fake_map[idx], 1))
        return samples

    allowed_idx = {idx for _, idx in allowed_ids}
    real_by_idx = {}
    for fp, parsed in zip(real_files, real_parsed):
        idx = parsed[1] if parsed is not None else _stem_id(fp)
        real_by_idx[idx] = fp

    fake_by_idx = {}
    for fp, parsed in zip(fake_files, fake_parsed):
        idx = parsed[1] if parsed is not None else _stem_id(fp)
        fake_by_idx[idx] = fp

    samples = []
    for idx in sorted(set(real_by_idx.keys()) & set(fake_by_idx.keys())):
        if idx not in allowed_idx:
            continue
        samples.append((real_by_idx[idx], 0))
        samples.append((fake_by_idx[idx], 1))
    return samples


def _collect_samples_unpaired(real_dirs, fake_dirs, allowed_ids):
    if not real_dirs or not fake_dirs:
        raise ValueError("real_dirs and fake_dirs must be non-empty lists.")

    real_files = []
    fake_files = []
    for d in real_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Real dataset directory not found: {d}")
        real_files.extend(glob.glob(os.path.join(d, "*.pt")))
    for d in fake_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Fake dataset directory not found: {d}")
        fake_files.extend(glob.glob(os.path.join(d, "*.pt")))

    if not real_files or not fake_files:
        raise FileNotFoundError("No .pt feature files found in real or fake dataset directories.")

    allowed_idx = {idx for _, idx in allowed_ids}

    # Filter real files by allowed_ids when possible
    real_samples = []
    for fp in real_files:
        parsed = _safe_parse_id(fp)
        if parsed is None:
            # UUID-style filenames: match by idx-only when possible.
            if _stem_id(fp) in allowed_idx:
                real_samples.append((fp, 0))
            continue
        if parsed in allowed_ids:
            real_samples.append((fp, 0))

    # For fake files, try to filter by allowed_ids when naming matches.
    fake_samples = []
    for fp in fake_files:
        parsed = _safe_parse_id(fp)
        if parsed is None:
            if _stem_id(fp) in allowed_idx:
                fake_samples.append((fp, 1))
            continue
        if parsed in allowed_ids:
            fake_samples.append((fp, 1))

    return real_samples + fake_samples


def _list_feature_files(dirs, side_name: str):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{side_name} dataset directory not found: {d}")
        files.extend(glob.glob(os.path.join(d, "*.pt")))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .pt feature files found in {side_name.lower()} dataset directories.")
    return files


def _collect_fake_samples_unpaired(fake_files, allowed_ids):
    allowed_idx = {idx for _, idx in allowed_ids}
    samples = []
    for fp in fake_files:
        parsed = _safe_parse_id(fp)
        if parsed is None:
            if _stem_id(fp) in allowed_idx:
                samples.append((fp, 1))
            continue
        if parsed in allowed_ids:
            samples.append((fp, 1))
    return samples


def _available_indices_from_dirs(dirs):
    idxs = set()
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Dataset directory not found: {d}")
        for fp in glob.glob(os.path.join(d, "*.pt")):
            parsed = _safe_parse_id(fp)
            idxs.add(parsed[1] if parsed is not None else _stem_id(fp))
    return idxs


def build_unpaired_split_loaders(
    split_cfg,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 99,
):
    """
    Build unpaired train/val/test loaders together so:
      1) each split has equal #real and #fake samples
      2) real samples do not overlap across splits
    """
    rng = random.Random(seed)
    split_order = [s for s in ("train", "val", "test") if s in split_cfg]
    if not split_order:
        raise ValueError("split_cfg must include at least one split among train/val/test.")

    fake_samples_by_split = {}
    real_candidates_by_split = {}
    for split in split_order:
        cfg = split_cfg[split]
        real_candidates_by_split[split] = _list_feature_files(cfg["real_dirs"], "Real")
        fake_files = _list_feature_files(cfg["fake_dirs"], "Fake")
        fake_samples = _collect_fake_samples_unpaired(fake_files, cfg["allowed_ids"])
        if not fake_samples:
            raise RuntimeError(f"No fake samples collected for split '{split}' in unpaired mode.")
        fake_samples_by_split[split] = fake_samples

    used_real = set()
    real_samples_by_split = {}
    for split in split_order:
        n_fake = len(fake_samples_by_split[split])
        candidates = [fp for fp in real_candidates_by_split[split] if fp not in used_real]
        if len(candidates) < n_fake:
            raise RuntimeError(
                f"Not enough non-overlapping real samples for split '{split}': "
                f"need {n_fake}, available {len(candidates)}"
            )
        chosen = rng.sample(candidates, n_fake) if len(candidates) > n_fake else candidates
        used_real.update(chosen)
        real_samples_by_split[split] = [(fp, 0) for fp in chosen]

    loaders = {}
    counts = {}
    for split in split_order:
        samples = real_samples_by_split[split] + fake_samples_by_split[split]
        ds = FeatureListDataset(samples)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=bool(split_cfg[split].get("shuffle", False)),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        counts[split] = {
            "real": len(real_samples_by_split[split]),
            "fake": len(fake_samples_by_split[split]),
            "total": len(samples),
        }
    return loaders, counts


def build_unpaired_split_loaders_fixed_counts(
    split_cfg,
    split_sizes,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 99,
):
    """
    Build unpaired loaders with fixed per-class counts for each split.
    Each split gets exactly split_sizes[split] real + split_sizes[split] fake samples.
    Sampling is deterministic under seed and avoids overlap within each class across splits.
    """
    rng = random.Random(seed)
    split_order = [s for s in ("train", "val", "test") if s in split_cfg and s in split_sizes]
    if not split_order:
        raise ValueError("split_cfg/split_sizes must include at least one split among train/val/test.")

    real_pool_by_split = {}
    fake_pool_by_split = {}
    for split in split_order:
        cfg = split_cfg[split]
        real_pool_by_split[split] = _list_feature_files(cfg["real_dirs"], "Real")
        fake_pool_by_split[split] = _list_feature_files(cfg["fake_dirs"], "Fake")

    used_real = set()
    used_fake = set()
    loaders = {}
    counts = {}
    for split in split_order:
        n = int(split_sizes[split])
        if n < 0:
            raise ValueError(f"split_sizes[{split}] must be >= 0, got {n}")

        real_candidates = [fp for fp in real_pool_by_split[split] if fp not in used_real]
        fake_candidates = [fp for fp in fake_pool_by_split[split] if fp not in used_fake]

        if len(real_candidates) < n:
            raise RuntimeError(
                f"Not enough real samples for split '{split}': need {n}, available {len(real_candidates)}"
            )
        if len(fake_candidates) < n:
            raise RuntimeError(
                f"Not enough fake samples for split '{split}': need {n}, available {len(fake_candidates)}"
            )

        chosen_real = rng.sample(real_candidates, n) if len(real_candidates) > n else real_candidates
        chosen_fake = rng.sample(fake_candidates, n) if len(fake_candidates) > n else fake_candidates
        used_real.update(chosen_real)
        used_fake.update(chosen_fake)

        samples = [(fp, 0) for fp in chosen_real] + [(fp, 1) for fp in chosen_fake]
        ds = FeatureListDataset(samples)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=bool(split_cfg[split].get("shuffle", False)),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        counts[split] = {
            "real": len(chosen_real),
            "fake": len(chosen_fake),
            "total": len(samples),
        }
    return loaders, counts


def build_data_loader_from_dirs(
    real_dirs,
    fake_dirs,
    allowed_ids,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle=False,
    unpaired: bool = False,
):
    if unpaired:
        samples = _collect_samples_unpaired(real_dirs, fake_dirs, allowed_ids)
    else:
        samples = _collect_samples(real_dirs, fake_dirs, allowed_ids)
        if not samples:
            print("[data] no paired samples found; falling back to unpaired collection.")
            samples = _collect_samples_unpaired(real_dirs, fake_dirs, allowed_ids)
    if not samples:
        raise RuntimeError("No samples collected for provided dataset dirs and ids")
    ds = FeatureListDataset(samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())

def build_test_loader_for_combo(
    real_dirs,
    fake_dirs,
    allowed_real_idx,
    allowed_fake_idx,
    batch_size: int = 128,
    num_workers: int = 4,
    unpaired: bool = False,
    target_per_class: int = None,
):
    real_files = []
    fake_files = []
    for d in real_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Real dataset directory not found: {d}")
        real_files.extend(glob.glob(os.path.join(d, "*.pt")))
    for d in fake_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Fake dataset directory not found: {d}")
        fake_files.extend(glob.glob(os.path.join(d, "*.pt")))
    if not real_files or not fake_files:
        raise FileNotFoundError("No .pt feature files found in real or fake dataset directories.")

    if unpaired:
        real_samples = []
        for fp in real_files:
            parsed = _safe_parse_id(fp)
            if parsed is None:
                if _stem_id(fp) in allowed_real_idx:
                    real_samples.append((fp, 0))
                continue
            if parsed[1] in allowed_real_idx:
                real_samples.append((fp, 0))

        fake_samples = []
        for fp in fake_files:
            parsed = _safe_parse_id(fp)
            if parsed is None:
                if _stem_id(fp) in allowed_fake_idx:
                    fake_samples.append((fp, 1))
                continue
            if parsed[1] in allowed_fake_idx:
                fake_samples.append((fp, 1))
        if not real_samples or not fake_samples:
            raise RuntimeError("Unpaired combo testing produced an empty real or fake pool.")

        n = min(len(real_samples), len(fake_samples))
        if target_per_class is not None:
            n = min(n, int(target_per_class))
        rng = random.Random(99)
        if len(real_samples) > n:
            real_samples = rng.sample(real_samples, n)
        if len(fake_samples) > n:
            fake_samples = rng.sample(fake_samples, n)
        samples = real_samples + fake_samples
    else:
        allowed_common_idx = set(allowed_real_idx) & set(allowed_fake_idx)
        cat_to_real = {}
        for fp in real_files:
            parsed = _safe_parse_id(fp)
            if parsed is None:
                continue
            cat, idx = parsed
            if idx in allowed_common_idx:
                cat_to_real.setdefault(cat, {})[idx] = fp

        cat_to_fake = {}
        for fp in fake_files:
            parsed = _safe_parse_id(fp)
            if parsed is None:
                continue
            cat, idx = parsed
            if idx in allowed_common_idx:
                cat_to_fake.setdefault(cat, {})[idx] = fp

        samples = []
        for cat, real_map in cat_to_real.items():
            fake_map = cat_to_fake.get(cat, {})
            for idx in sorted(set(real_map.keys()) & set(fake_map.keys())):
                samples.append((real_map[idx], 0))
                samples.append((fake_map[idx], 1))
        if target_per_class is not None:
            real_samples = [s for s in samples if s[1] == 0]
            fake_samples = [s for s in samples if s[1] == 1]
            n = min(len(real_samples), len(fake_samples), int(target_per_class))
            rng = random.Random(99)
            if len(real_samples) > n:
                real_samples = rng.sample(real_samples, n)
            if len(fake_samples) > n:
                fake_samples = rng.sample(fake_samples, n)
            samples = real_samples + fake_samples

    if not samples:
        raise RuntimeError("No samples collected for combo test loader.")

    ds = FeatureListDataset(samples)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def infer_feature_dim(real_dirs, fake_dirs) -> int:
    """
    Inspect the first available feature tensor in the dataset to determine its dimensionality.
    """
    for d in (real_dirs or []) + (fake_dirs or []):
        files = sorted(glob.glob(os.path.join(d, "*.pt")))
        if not files:
            continue
        feat = _load_feature_tensor(files[0])
        if feat.dim() != 1:
            raise ValueError(f"Expected 1D feature tensor, got shape {tuple(feat.shape)} in {files[0]}")
        return feat.numel()
    raise FileNotFoundError("Could not find .pt feature files under any provided dataset directory")


def evaluate_loader_like_test(model: Model, loader: DataLoader, device: torch.device, threshold: float = 0.5):
    model.eval()
    failure_count = 0
    with torch.no_grad():
        y_true, y_pred = [], []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            _, pred = model(x)
            scores = pred.sigmoid().view(-1)
            preds = (scores > threshold).float()

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(scores.detach().cpu().numpy())

            mismatched = preds.ne(y).detach().cpu().numpy().astype(bool)
            if mismatched.any():
                y_cpu = y.detach().cpu().numpy()
                for i, bad in enumerate(mismatched):
                    if bad and int(y_cpu[i]) == 1:
                        failure_count += 1

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)

    ap = average_precision_score(y_true, y_pred)
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)
    r_acc = accuracy_score(y_true[real_mask], (y_pred[real_mask] > threshold)) if real_mask.any() else float("nan")
    f_acc = accuracy_score(y_true[fake_mask], (y_pred[fake_mask] > threshold)) if fake_mask.any() else float("nan")
    acc = accuracy_score(y_true, (y_pred > threshold))
    return ap, r_acc, f_acc, acc, failure_count


def evaluate_all_checkpoints(
    base_opt,
    run_dirs,
    device,
    test_loader,
    dataset_name=None,
    result_tag=None,
    combined_mode: bool = False,
    reset_combined: bool = False,
):
    """
    After training, evaluate all saved checkpoints in every run directory on the provided test split.
    Writes:
      - results_<checkpoint>.txt   (per run directory)
      - summary.csv (per run directory with mACC/mAP per checkpoint)
    """
    if test_loader is None:
        print("[eval] No test loader provided — skipping evaluation.")
        return

    dataset_name = dataset_name or os.path.basename(os.path.normpath(base_opt.dataset))
    threshold = float(getattr(base_opt, "threshold", 0.5))
    feature_dim = getattr(base_opt, "feature_dim", 512)
    safe_tag = _sanitize_result_tag(result_tag) if result_tag else None

    for run_dir in run_dirs:
        escaped_run_dir = glob.escape(run_dir)
        # find checkpoints
        ckpts = sorted(glob.glob(os.path.join(escaped_run_dir, "model_epoch_*.pth")))
        ckpts += sorted(glob.glob(os.path.join(escaped_run_dir, "model_*_epoch_*.pth")))
        ckpts = list(dict.fromkeys(ckpts))
        best_path = os.path.join(run_dir, "model_epoch_best.pth")
        if os.path.isfile(best_path):
            ckpts = [best_path] + [p for p in ckpts if p != best_path]  # evaluate "best" first

        summary_rows = []
        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"[eval] {os.path.basename(run_dir)} -> {ckpt_name}")

            if combined_mode and safe_tag:
                out_txt = os.path.join(run_dir, f"results_{ckpt_name}.txt")
            elif safe_tag:
                out_txt = os.path.join(run_dir, f"results_{safe_tag}_{ckpt_name}.txt")
            else:
                out_txt = os.path.join(run_dir, f"results_{ckpt_name}.txt")
            if (not combined_mode) and os.path.exists(out_txt):
                print(f"[eval] results already exist, skipping: {out_txt}")
                continue

            # load model
            model = Model(feature_dim=feature_dim).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])

            ap, r_acc, f_acc, acc, failures = evaluate_loader_like_test(
                model, test_loader, device, threshold=threshold
            )
            if combined_mode and safe_tag:
                mode = "a"
                if reset_combined and not os.path.exists(out_txt):
                    mode = "w"
                with open(out_txt, mode) as f:
                    f.write(f"[{safe_tag}] {dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f}\n")
                    f.write(f"r_acc={r_acc*100:.2f} f_acc={f_acc*100:.2f} threshold={threshold}\n")
                    f.write(f"failures={failures}\n\n")
            else:
                with open(out_txt, "w") as f:
                    f.write(f"{dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f}\n")
                    f.write(f"r_acc={r_acc*100:.2f} f_acc={f_acc*100:.2f} threshold={threshold}\n")
                    f.write(f"failures={failures}\n")

            # add to summary
            summary_rows.append([
                ckpt_name,
                f"{acc*100:.3f}",
                f"{ap*100:.3f}",
                f"{r_acc*100:.3f}",
                f"{f_acc*100:.3f}",
                str(failures),
            ])

            print(
                f"[eval] {dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f} "
                f"r_acc={r_acc*100:.2f} f_acc={f_acc*100:.2f}"
            )

            # free GPU RAM
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # write summary csv
        if summary_rows and (not combined_mode):
            summary_name = f"summary_{safe_tag}.csv" if safe_tag else "summary.csv"
            with open(os.path.join(run_dir, summary_name), "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["checkpoint", "acc", "ap", "r_acc", "f_acc", "failures"])
                writer.writerows(summary_rows)


def _get_checkpoints_in_run_dir(run_dir):
    escaped_run_dir = glob.escape(run_dir)
    ckpts = sorted(glob.glob(os.path.join(escaped_run_dir, "model_epoch_*.pth")))
    ckpts += sorted(glob.glob(os.path.join(escaped_run_dir, "model_*_epoch_*.pth")))
    ckpts = list(dict.fromkeys(ckpts))
    best_ckpt = os.path.join(run_dir, "model_epoch_best.pth")
    if os.path.isfile(best_ckpt):
        ckpts = [best_ckpt] + [p for p in ckpts if p != best_ckpt]
    return ckpts


def _filter_run_dirs_pending_eval(run_dirs):
    pending = []
    for run_dir in run_dirs:
        ckpts = _get_checkpoints_in_run_dir(run_dir)
        if not ckpts:
            print(f"[eval-10] no checkpoints found in {run_dir}")
            continue

        has_pending = False
        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            out_txt = os.path.join(run_dir, f"results_{ckpt_name}.txt")
            if not os.path.exists(out_txt):
                has_pending = True
                break

        if has_pending:
            pending.append(run_dir)
        else:
            print(f"[eval-10] all checkpoints already tested in {run_dir}; skipping run.")
    return pending


def evaluate_checkpoints_on_10_combos(base_opt, run_dirs, device, test_ids):
    threshold = float(getattr(base_opt, "threshold", 0.5))
    feature_dim = getattr(base_opt, "feature_dim", 512)
    if isinstance(test_ids, dict):
        split_ids = test_ids
    else:
        split_ids = {"test": test_ids, "train": set(), "val": set()}
    test_idx = {idx for _, idx in split_ids["test"]}
    forbidden_idx = {idx for _, idx in (split_ids["train"] | split_ids["val"])}

    target_per_class = None
    combo_specs = []
    for real_ds, fake_ds in TEN_TEST_COMBOS:
        combo_tag = f"real_{real_ds}__fake_{fake_ds}"
        use_unpaired = (real_ds, fake_ds) not in PAIRED_TEST_COMBOS
        dataset_name = f"real[{real_ds}]_fake[{fake_ds}]"

        real_dirs = _resolve_dataset_paths([real_ds], base_opt.pc_root, base_opt.pc_variant, base_opt.pc_backbone)
        fake_dirs = _resolve_dataset_paths([fake_ds], base_opt.pc_root, base_opt.pc_variant, base_opt.pc_backbone)
        _validate_variant_backbone(real_dirs, base_opt.pc_variant, base_opt.pc_backbone)
        _validate_variant_backbone(fake_dirs, base_opt.pc_variant, base_opt.pc_backbone)

        real_idx_all = _available_indices_from_dirs(real_dirs)
        fake_idx_all = _available_indices_from_dirs(fake_dirs)
        if use_unpaired:
            allowed_real_idx = real_idx_all & test_idx
            allowed_fake_idx = fake_idx_all & test_idx
            if not allowed_real_idx:
                allowed_real_idx = real_idx_all - forbidden_idx
            if not allowed_fake_idx:
                allowed_fake_idx = fake_idx_all - forbidden_idx
        else:
            common_idx = real_idx_all & fake_idx_all
            allowed_common = common_idx & test_idx
            if not allowed_common:
                allowed_common = common_idx - forbidden_idx
            allowed_real_idx = allowed_common
            allowed_fake_idx = allowed_common

        combo_loader = build_test_loader_for_combo(
            real_dirs,
            fake_dirs,
            allowed_real_idx,
            allowed_fake_idx,
            batch_size=getattr(base_opt, "batch_size", 128),
            num_workers=getattr(base_opt, "num_workers", 4),
            unpaired=use_unpaired,
            target_per_class=target_per_class,
        )

        if target_per_class is None:
            target_per_class = len(combo_loader.dataset) // 2
        print(
            f"[eval-10] combo real={real_ds} fake={fake_ds} "
            f"mode={'unpaired' if use_unpaired else 'paired'} size={len(combo_loader.dataset)} "
            f"(per_class={target_per_class})"
        )
        combo_specs.append((combo_tag, dataset_name, combo_loader))

    for run_dir in run_dirs:
        ckpts = _get_checkpoints_in_run_dir(run_dir)
        if not ckpts:
            print(f"[eval-10] no checkpoints found in {run_dir}")
            continue

        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            out_txt = os.path.join(run_dir, f"results_{ckpt_name}.txt")
            print(f"[eval-10] checkpoint {os.path.basename(run_dir)} -> {ckpt_name}")
            if os.path.exists(out_txt):
                print(f"[eval-10] results already exist, skipping: {out_txt}")
                continue

            model = Model(feature_dim=feature_dim).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])

            lines = []
            for combo_tag, dataset_name, combo_loader in combo_specs:
                ap, r_acc, f_acc, acc, failures = evaluate_loader_like_test(
                    model, combo_loader, device, threshold=threshold
                )
                lines.append(f"{dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f}")
                lines.append(f"r_acc={r_acc*100:.2f} f_acc={f_acc*100:.2f} threshold={threshold}")
                lines.append(f"failures={failures}")
                lines.append("")

            while lines and lines[-1] == "":
                lines.pop()
            with open(out_txt, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[eval-10] wrote combined report: {out_txt}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ------------- main -------------
if __name__ == '__main__':
    # Parse once (defer printing until after feature dim is inferred)
    opt_parser = TrainOptions()
    base_opt = opt_parser.parse(print_options=False)

    train_real = _parse_csv_list(getattr(base_opt, "train_real_datasets", ""))
    train_fake = _parse_csv_list(getattr(base_opt, "train_fake_datasets", ""))
    if not train_real or not train_fake:
        raise ValueError("--train_real_datasets and --train_fake_datasets are required.")

    train_real_dirs = _resolve_dataset_paths(train_real, base_opt.pc_root, base_opt.pc_variant, base_opt.pc_backbone)
    train_fake_dirs = _resolve_dataset_paths(train_fake, base_opt.pc_root, base_opt.pc_variant, base_opt.pc_backbone)

    _validate_variant_backbone(train_real_dirs, base_opt.pc_variant, base_opt.pc_backbone)
    _validate_variant_backbone(train_fake_dirs, base_opt.pc_variant, base_opt.pc_backbone)

    train_real_names = [_dataset_name_from_path(d) for d in train_real_dirs]
    train_fake_names = [_dataset_name_from_path(d) for d in train_fake_dirs]
    train_code = _dataset_group_abbreviation(train_real_names) + _dataset_group_abbreviation(train_fake_names)
    train_label = f"train[{train_code}]"
    dataset_label = train_label
    base_opt.dataset = dataset_label
    base_opt.name = f"{base_opt.name}_{dataset_label}_{base_opt.pc_variant}_{base_opt.pc_backbone}"

    split_ids = {
        "train": _load_ids(base_opt.train_ids),
        "val": _load_ids(base_opt.val_ids),
        "test": _load_ids(base_opt.test_ids),
    }
    _assert_split_ids_disjoint(split_ids)

    # Infer feature dimensionality from dataset to avoid shape mismatches
    inferred_dim = infer_feature_dim(train_real_dirs, train_fake_dirs)
    if getattr(base_opt, "feature_dim", None) != inferred_dim:
        prev = getattr(base_opt, "feature_dim", "unset")
        print(f"[data] inferred feature_dim={inferred_dim} from {dataset_label}; overriding {prev}")
        base_opt.feature_dim = inferred_dim
    opt_parser.print_options(base_opt)

    # Device
    device = torch.device(
        f'cuda:{base_opt.gpu_ids[0]}' if (getattr(base_opt, "gpu_ids", None) and torch.cuda.is_available())
        else 'cpu'
    )

    # Load TRAIN/VAL splits from feature folders
    auto_unpaired = _should_auto_unpaired(train_real_names, train_fake_names)
    if auto_unpaired:
        split_cfg = {
            "train": {
                "real_dirs": train_real_dirs,
                "fake_dirs": train_fake_dirs,
                "allowed_ids": split_ids["train"],
                "shuffle": True,
            },
            "val": {
                "real_dirs": train_real_dirs,
                "fake_dirs": train_fake_dirs,
                "allowed_ids": split_ids["val"],
                "shuffle": False,
            },
            "test": {
                "real_dirs": train_real_dirs,
                "fake_dirs": train_fake_dirs,
                "allowed_ids": split_ids["test"],
                "shuffle": False,
            },
        }
        split_sizes = {k: len(v) for k, v in split_ids.items()}
        loaders, counts = build_unpaired_split_loaders_fixed_counts(
            split_cfg,
            split_sizes=split_sizes,
            seed=99,
        )
        print(
            f"[data] auto-unpaired sampling enabled (real={train_real_names}, fake={train_fake_names}); "
            f"fixed split sizes with seed=99: "
            f"train={split_sizes['train']} val={split_sizes['val']} test={split_sizes['test']}"
        )
        print(
            f"[data] auto-unpaired loaders: "
            f"train={counts['train']['total']} (real={counts['train']['real']} fake={counts['train']['fake']}) "
            f"val={counts['val']['total']} (real={counts['val']['real']} fake={counts['val']['fake']}) "
            f"test={counts['test']['total']} (real={counts['test']['real']} fake={counts['test']['fake']})"
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
    else:
        train_loader = build_data_loader_from_dirs(
            train_real_dirs, train_fake_dirs, split_ids["train"], shuffle=True, unpaired=False
        )
        val_loader = build_data_loader_from_dirs(
            train_real_dirs, train_fake_dirs, split_ids["val"], shuffle=False, unpaired=False
        )
        print(
            f"[data] loaders: train={len(train_loader.dataset)} "
            f"val={len(val_loader.dataset)}"
        )

    # Build grids
    lambda_x_grid = _parse_int_list(base_opt.lambda_x_list)
    lambda_y_grid = _parse_int_list(base_opt.lambda_y_list)

    # Keep track of run directories for later evaluation
    created_run_dirs = []
    base_run_dir = os.path.join(base_opt.checkpoints_dir, base_opt.name)
    os.makedirs(base_run_dir, exist_ok=True)

    # Optional eval-only mode
    if getattr(base_opt, "eval_only", False):
        pattern = os.path.join(base_opt.checkpoints_dir, f"{glob.escape(base_opt.name)}*")
        run_roots = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        run_dirs = []
        for root in run_roots:
            combo_pattern = os.path.join(glob.escape(root), "lx*_ly*")
            combo_dirs = [d for d in glob.glob(combo_pattern) if os.path.isdir(d)]
            if combo_dirs:
                run_dirs.extend(sorted(combo_dirs))
            else:
                run_dirs.append(root)
        if not run_dirs:
            raise FileNotFoundError(f"Eval-only requested but no run directories matched pattern: {pattern}")
        run_dirs_to_eval = _filter_run_dirs_pending_eval(run_dirs)
        if not run_dirs_to_eval:
            print("\n=== Eval-only: all checkpoints already have results_*.txt; nothing to evaluate ===")
            raise SystemExit(0)
        print(f"\n=== Eval-only: evaluating checkpoints under: {run_dirs_to_eval} on 10 combos ===")
        evaluate_checkpoints_on_10_combos(base_opt, run_dirs_to_eval, device, split_ids)
        raise SystemExit(0)

    # Sweep training (no validation, no test yet)
    for lx in lambda_x_grid:
        for ly in lambda_y_grid:
            run_opt = deepcopy(base_opt)
            run_opt.lambda_x = lx
            run_opt.lambda_y = ly
            run_opt.name = os.path.join(base_opt.name, f"lx{lx}_ly{ly}")
            run_opt.run_tag = f"lx{lx}_ly{ly}"

            out_dir = os.path.join(run_opt.checkpoints_dir, run_opt.name)
            os.makedirs(out_dir, exist_ok=True)
            created_run_dirs.append(out_dir)

            # Skip if final checkpoint already exists (assumes 0-based epochs)
            epochs = getattr(base_opt, "epochs", None)
            last_epoch = (epochs - 1) if isinstance(epochs, int) and epochs > 0 else 99
            final_ckpt = os.path.join(out_dir, f"model_{run_opt.run_tag}_epoch_{last_epoch}.pth")
            if os.path.isfile(final_ckpt):
                print(f"[skip] Found {os.path.basename(final_ckpt)} in {out_dir}; skipping training.")
                continue

            print(f"\n=== Training: dataset={dataset_label}  λx={lx}  λy={ly}  -> {run_opt.name} ===")
            trainer = Trainer(run_opt, train_loader, feature_dim=getattr(base_opt, "feature_dim", 512))  # only train loader
            trainer.run()  # saves model_epoch_{E}.pth (and model_epoch_best if you keep logic)
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # After all training finishes, evaluate checkpoints on the fixed 10 test combos.
    run_dirs_to_eval = _filter_run_dirs_pending_eval(created_run_dirs)
    if not run_dirs_to_eval:
        print("\n=== All training finished. All checkpoints already have results_*.txt; skipping evaluation ===")
    else:
        print("\n=== All training finished. Starting evaluation on 10 test combos ===")
        evaluate_checkpoints_on_10_combos(base_opt, run_dirs_to_eval, device, split_ids)
