from copy import deepcopy
import os
import glob
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, accuracy_score
import optuna

from options.base_options import BaseOptions
from networks.trainer_vib import VIBTrainer
from networks.base_model import VIB
from train_hsic import (
    PAIRED_TEST_COMBOS,
    TEN_TEST_COMBOS,
    _assert_split_ids_disjoint,
    _available_indices_from_dirs,
    _dataset_group_abbreviation,
    _dataset_name_from_path,
    _load_ids,
    _parse_csv_list,
    _resolve_dataset_paths,
    _should_auto_unpaired,
    _validate_variant_backbone,
    build_data_loader_from_dirs,
    build_test_loader_for_combo,
    build_unpaired_split_loaders_fixed_counts,
    infer_feature_dim,
)


# ------------- options -------------
class VIBOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--n_trials', type=int, default=100,
                            help='number of Optuna trials to run')
        parser.add_argument('--beta_min', type=float, default=5e-4,
                            help='minimum beta (inclusive) for log-uniform search')
        parser.add_argument('--beta_max', type=float, default=1e-1,
                            help='maximum beta (inclusive) for log-uniform search')
        parser.add_argument('--lr_min', type=float, default=1e-5,
                            help='minimum learning rate (inclusive) for log-uniform search')
        parser.add_argument('--lr_max', type=float, default=1e-2,
                            help='maximum learning rate (inclusive) for log-uniform search')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='DataLoader workers; set to 0 if multiprocessing is restricted')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='DataLoader batch size')
        return parser


def compute_scores_vib(model: VIB, loader: DataLoader, device: torch.device):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)
            _, pred = model(x, mode='val')
            y_pred.append(pred.sigmoid().detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)

    ap = average_precision_score(y_true, y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], (y_pred[y_true == 0] > 0.5))
    f_acc = accuracy_score(y_true[y_true == 1], (y_pred[y_true == 1] > 0.5))
    acc = accuracy_score(y_true, (y_pred > 0.5))
    return ap, r_acc, f_acc, acc


def evaluate_checkpoints(run_dirs, device, test_loader, dataset_name, feature_dim: int):
    for run_dir in run_dirs:
        ckpts = sorted(glob.glob(os.path.join(run_dir, "model_epoch_*.pth")))
        if not ckpts:
            continue

        summary_rows = []
        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            print(f"[eval] {os.path.basename(run_dir)} -> {ckpt_name}")

            model = VIB(feature_dim=feature_dim).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])

            lines = []
            acc_vals, ap_vals = [], []
            ap, r_acc, f_acc, acc = compute_scores_vib(model, test_loader, device)
            lines.append(f"{dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f}")
            acc_vals.append(acc * 100)
            ap_vals.append(ap * 100)

            mACC = float(np.mean(acc_vals)) if acc_vals else float("nan")
            mAP = float(np.mean(ap_vals)) if ap_vals else float("nan")
            lines.append(f"mACC={mACC:.2f} mAP={mAP:.2f}")

            out_txt = os.path.join(run_dir, f"results_{ckpt_name}.txt")
            with open(out_txt, "w") as f:
                f.write("\n".join(lines) + "\n")

            summary_rows.append([ckpt_name, f"{mACC:.3f}", f"{mAP:.3f}"])

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if summary_rows:
            with open(os.path.join(run_dir, "summary.csv"), "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["checkpoint", "mACC", "mAP"])
                writer.writerows(summary_rows)


def evaluate_checkpoints_on_10_combos(base_opt, run_dirs, device, split_ids, feature_dim: int):
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
        ckpts = sorted(glob.glob(os.path.join(run_dir, "model_epoch_*.pth")))
        if not ckpts:
            print(f"[eval-10] no checkpoints found in {run_dir}")
            continue

        summary_rows = []
        for ckpt_path in ckpts:
            ckpt_name = os.path.basename(ckpt_path)
            out_txt = os.path.join(run_dir, f"results_10_combos_{ckpt_name}.txt")
            print(f"[eval-10] checkpoint {os.path.basename(run_dir)} -> {ckpt_name}")
            if os.path.exists(out_txt):
                print(f"[eval-10] results already exist, skipping: {out_txt}")
                continue

            model = VIB(feature_dim=feature_dim).to(device)
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state["model"])

            lines = []
            acc_vals, ap_vals = [], []
            for combo_tag, dataset_name, combo_loader in combo_specs:
                ap, r_acc, f_acc, acc = compute_scores_vib(model, combo_loader, device)
                lines.append(f"{dataset_name}: acc={acc*100:.2f} ap={ap*100:.2f}")
                lines.append(f"r_acc={r_acc*100:.2f} f_acc={f_acc*100:.2f}")
                lines.append("")
                acc_vals.append(acc * 100)
                ap_vals.append(ap * 100)

            mACC = float(np.mean(acc_vals)) if acc_vals else float("nan")
            mAP = float(np.mean(ap_vals)) if ap_vals else float("nan")
            lines.append(f"mACC={mACC:.2f} mAP={mAP:.2f}")
            with open(out_txt, "w") as f:
                f.write("\n".join(lines) + "\n")
            summary_rows.append([ckpt_name, f"{mACC:.3f}", f"{mAP:.3f}"])
            print(f"[eval-10] wrote combined report: {out_txt}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if summary_rows:
            with open(os.path.join(run_dir, "summary_10_combos.csv"), "w", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["checkpoint", "mACC", "mAP"])
                writer.writerows(summary_rows)


# ------------- main -------------
if __name__ == '__main__':
    opt_parser = VIBOptions()
    opt = opt_parser.parse(print_options=False)

    train_real = _parse_csv_list(getattr(opt, "train_real_datasets", ""))
    train_fake = _parse_csv_list(getattr(opt, "train_fake_datasets", ""))
    if not train_real or not train_fake:
        raise ValueError("--train_real_datasets and --train_fake_datasets are required.")

    train_real_dirs = _resolve_dataset_paths(train_real, opt.pc_root, opt.pc_variant, opt.pc_backbone)
    train_fake_dirs = _resolve_dataset_paths(train_fake, opt.pc_root, opt.pc_variant, opt.pc_backbone)
    _validate_variant_backbone(train_real_dirs, opt.pc_variant, opt.pc_backbone)
    _validate_variant_backbone(train_fake_dirs, opt.pc_variant, opt.pc_backbone)

    train_real_names = [_dataset_name_from_path(d) for d in train_real_dirs]
    train_fake_names = [_dataset_name_from_path(d) for d in train_fake_dirs]
    train_code = _dataset_group_abbreviation(train_real_names) + _dataset_group_abbreviation(train_fake_names)
    dataset_label = f"train[{train_code}]"
    opt.dataset = dataset_label
    opt.name = f"{opt.name}_{dataset_label}_{opt.pc_variant}_{opt.pc_backbone}"

    split_ids = {
        "train": _load_ids(opt.train_ids),
        "val": _load_ids(opt.val_ids),
        "test": _load_ids(opt.test_ids),
    }
    _assert_split_ids_disjoint(split_ids)

    inferred_dim = infer_feature_dim(train_real_dirs, train_fake_dirs)
    if getattr(opt, "feature_dim", None) != inferred_dim:
        prev = getattr(opt, "feature_dim", "unset")
        print(f"[data] inferred feature_dim={inferred_dim} from {dataset_label}; overriding {prev}")
        opt.feature_dim = inferred_dim
    opt_parser.print_options(opt)

    device = torch.device(
        f'cuda:{opt.gpu_ids[0]}' if (getattr(opt, "gpu_ids", None) and torch.cuda.is_available())
        else 'cpu'
    )

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
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            seed=99,
        )
        print(
            f"[data] auto-unpaired loaders: "
            f"train={counts['train']['total']} (real={counts['train']['real']} fake={counts['train']['fake']}) "
            f"val={counts['val']['total']} (real={counts['val']['real']} fake={counts['val']['fake']}) "
            f"test={counts['test']['total']} (real={counts['test']['real']} fake={counts['test']['fake']})"
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
    else:
        train_loader = build_data_loader_from_dirs(
            train_real_dirs,
            train_fake_dirs,
            split_ids["train"],
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            unpaired=False,
        )
        val_loader = build_data_loader_from_dirs(
            train_real_dirs,
            train_fake_dirs,
            split_ids["val"],
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=False,
            unpaired=False,
        )
        test_loader = build_data_loader_from_dirs(
            train_real_dirs,
            train_fake_dirs,
            split_ids["test"],
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=False,
            unpaired=False,
        )
    print(f"[data] loaders: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    dataset_name = dataset_label

    run_dirs = []

    def objective(trial: optuna.Trial):
        beta = trial.suggest_float("beta", opt.beta_min, opt.beta_max, log=True)
        lr = trial.suggest_float("lr", opt.lr_min, opt.lr_max, log=True)
        run_opt = deepcopy(opt)
        run_opt.name = f"{opt.name}_trial{trial.number}_beta{beta:.6g}_lr{lr:.2e}"

        out_dir = os.path.join(run_opt.checkpoints_dir, run_opt.name)
        os.makedirs(out_dir, exist_ok=True)
        run_dirs.append(out_dir)
        trial.set_user_attr("run_dir", out_dir)

        with open(os.path.join(out_dir, 'hparams.txt'), 'w') as f:
            f.write(f"dataset={opt.dataset}\nbeta={beta}\nlr={lr}\nfeature_dim={opt.feature_dim}\n")

        print(f"\n=== Training VIB (trial {trial.number}): dataset={opt.dataset}  beta={beta}  lr={lr}  -> {run_opt.name} ===")
        trainer = VIBTrainer(run_opt, train_loader, beta=beta, lr=lr, feature_dim=opt.feature_dim)
        trainer.run()
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate last checkpoint on validation split (use final epoch)
        ckpts = sorted(glob.glob(os.path.join(out_dir, "model_epoch_*.pth")))
        if not ckpts:
            raise RuntimeError(f"No checkpoints saved in {out_dir}")
        last_ckpt = ckpts[-1]
        model = VIB(feature_dim=opt.feature_dim).to(device)
        state = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        ap, r_acc, f_acc, acc = compute_scores_vib(model, val_loader, device)
        trial.set_user_attr("val_metrics", {"ap": ap, "acc": acc, "r_acc": r_acc, "f_acc": f_acc, "beta": beta, "lr": lr})
        # maximize mACC (average of real/fake accuracies)
        macc = (r_acc + f_acc) / 2.0
        return macc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=opt.n_trials)

    best_trial = study.best_trial
    best_run_dir = best_trial.user_attrs.get("run_dir")
    print("\n=== Optuna finished ===")
    print(f"Best trial: {best_trial.number}  beta={best_trial.params['beta']:.6g}  lr={best_trial.params['lr']:.2e}  mACC={best_trial.value:.4f}")
    if "val_metrics" in best_trial.user_attrs:
        vm = best_trial.user_attrs["val_metrics"]
        print(f"Validation metrics: acc={vm['acc']:.4f} ap={vm['ap']:.4f} r_acc={vm['r_acc']:.4f} f_acc={vm['f_acc']:.4f}")

    if best_run_dir is not None:
        print("\n=== Evaluating best run on test split ===")
        evaluate_checkpoints([best_run_dir], device, test_loader, dataset_name, feature_dim=opt.feature_dim)
        print("\n=== Evaluating best run on 10 test combos ===")
        evaluate_checkpoints_on_10_combos(opt, [best_run_dir], device, split_ids, feature_dim=opt.feature_dim)
    else:
        print("No best run directory recorded; skipping test evaluation.")
