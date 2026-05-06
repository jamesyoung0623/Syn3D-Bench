# Syn3D-Bench

This repository trains feature-based synthetic-3D detectors with two training
entry points:

- `train_hsic.py`: HSIC Bottleneck detector
- `train_vib.py`: VIB-Net detector

Both methods train ShapeNet as the real dataset against one generated fake
dataset, using precomputed point-cloud feature tensors.

## Setup

Run commands from the repository root because the training scripts use relative
paths such as `datasets/PCs`, `train_ids.txt`, `val_ids.txt`, and
`test_ids.txt`.

```bash
cd /path/to/Syn3D-Bench
```

Create and activate a Conda environment. Python 3.10 is recommended.

```bash
conda create -n syn3d-bench python=3.10 -y
conda activate syn3d-bench
```

Install PyTorch for your CUDA version. For example, CUDA 11.7:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
```

If you do not need CUDA, install CPU PyTorch instead:

```bash
pip install torch==1.13.1 torchvision==0.14.1
```

Install the remaining dependencies:

```bash
pip install numpy scikit-learn optuna tensorboardX huggingface_hub
```

Each wrapper trains the same six point-cloud feature settings:

- `ULIP-2` with `pointnext`
- `ULIP-2` with `pointbert_xyz`
- `ULIP-1` with `pointbert`
- `ULIP-1` with `pointnext`
- `ULIP-1` with `pointmlp`
- `ULIP-1` with `pointnet2_ssg`

## HSIC Bottleneck

HSIC Bottleneck training uses `train_hsic.py`. For each point-cloud setting, it
runs a grid over `--lambda_x_list` and `--lambda_y_list`. If you do not override
them, both lists default to:

```text
0,100,200,300,400,500,600,700,800,900,1000
```

That default is 121 lambda combinations for each point-cloud setting.

### HSIC Wrappers

```bash
cd /path/to/Syn3D-Bench
./run_train_instantmesh_hsic.sh
./run_train_lgm_hsic.sh
./run_train_sam3d_hsic.sh
./run_train_trellis_hsic.sh
./run_train_trellis_text_hsic.sh
```

GPU assignments:

- GPU 0: `run_train_instantmesh_hsic.sh`, `run_train_sam3d_hsic.sh`,
  `run_train_trellis_hsic.sh`
- GPU 1: `run_train_lgm_hsic.sh`, `run_train_trellis_text_hsic.sh`

Each HSIC wrapper calls:

```bash
python train_hsic.py \
  --train_real_datasets Shapenet \
  --train_fake_datasets <FAKE_DATASET> \
  --pc_variant <ULIP-1-or-ULIP-2> \
  --pc_backbone <BACKBONE> \
  --gpu_ids <GPU>
```

Single HSIC run with a reduced lambda grid:

```bash
cd /path/to/Syn3D-Bench
python train_hsic.py \
  --train_real_datasets Shapenet \
  --train_fake_datasets InstantMesh \
  --pc_variant ULIP-2 \
  --pc_backbone pointnext \
  --lambda_x_list 500 \
  --lambda_y_list 300 \
  --gpu_ids 0
```

Evaluate existing HSIC checkpoints without training:

```bash
cd /path/to/Syn3D-Bench
python train_hsic.py \
  --train_real_datasets Shapenet \
  --train_fake_datasets InstantMesh \
  --pc_variant ULIP-2 \
  --pc_backbone pointnext \
  --eval_only \
  --gpu_ids 0
```

HSIC outputs are written to `checkpoints/`. Lambda-grid runs are saved inside
`lx*_ly*` subdirectories:

```text
checkpoints/HSIC_grid_train[SI]_ULIP-2_pointnext/
checkpoints/HSIC_grid_train[SI]_ULIP-2_pointnext/lx500_ly300/
```

Each lambda run directory contains:

- `model_lx<LX>_ly<LY>_epoch_<N>.pth`
- TensorBoard event files with `loss`, `loss_task`, and `loss_bottle`
- `results_model_lx<LX>_ly<LY>_epoch_<N>.pth.txt`

## VIB-Net

VIB-Net training uses `train_vib.py`. It runs Optuna over VIB `beta` and
learning rate. Defaults:

- `--n_trials 5`
- `--beta_min 5e-4`
- `--beta_max 1e-1`
- `--lr_min 1e-5`
- `--lr_max 1e-2`

### VIB Wrappers

```bash
cd /path/to/Syn3D-Bench
./run_train_instantmesh_vib.sh
./run_train_lgm_vib.sh
./run_train_sam3d_vib.sh
./run_train_trellis_vib.sh
./run_train_trellis_text_vib.sh
```

GPU assignments:

- GPU 0: `run_train_instantmesh_vib.sh`, `run_train_sam3d_vib.sh`,
  `run_train_trellis_vib.sh`
- GPU 1: `run_train_lgm_vib.sh`, `run_train_trellis_text_vib.sh`

Each VIB wrapper calls:

```bash
python train_vib.py \
  --name VIB \
  --train_real_datasets Shapenet \
  --train_fake_datasets <FAKE_DATASET> \
  --pc_variant <ULIP-1-or-ULIP-2> \
  --pc_backbone <BACKBONE> \
  --gpu_ids <GPU>
```

Single VIB run with a smaller Optuna search:

```bash
cd /path/to/Syn3D-Bench
python train_vib.py \
  --name VIB \
  --train_real_datasets Shapenet \
  --train_fake_datasets InstantMesh \
  --pc_variant ULIP-2 \
  --pc_backbone pointnext \
  --n_trials 2 \
  --beta_min 0.001 \
  --beta_max 0.01 \
  --lr_min 0.0001 \
  --lr_max 0.001 \
  --gpu_ids 0
```

VIB outputs are written to `checkpoints/`. Trial directories include the
training combo, point-cloud setting, trial number, beta, and learning rate:

```text
checkpoints/VIB_train[SI]_ULIP-2_pointnext_trial0_beta..._lr.../
```

Each VIB trial directory contains:

- `hparams.txt`
- `model_epoch_<N>.pth`
- TensorBoard logs under `train/`
- `results_model_epoch_<N>.pth.txt` for evaluated checkpoints
- `summary.csv` for same-pair held-out test evaluation
- `results_10_combos_model_epoch_<N>.pth.txt` for 10-combo evaluation
- `summary_10_combos.csv` for 10-combo `mACC` and `mAP`

## Download Dataset

The Syn3D feature dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/jamesyoung0623/Syn3D-Dataset
```

Install the Hugging Face Hub CLI if needed:

```bash
pip install -U huggingface_hub
```

Download the dataset into the repository `datasets/` directory:

```bash
cd /path/to/Syn3D-Bench
mkdir -p datasets
huggingface-cli download jamesyoung0623/Syn3D-Dataset \
  --repo-type dataset \
  --local-dir datasets
```

After download, confirm the point-cloud feature tree exists:

```bash
ls datasets/PCs
```

The training scripts expect paths such as:

```text
datasets/PCs/Shapenet/ULIP-2/pointnext/*.pt
datasets/PCs/InstantMesh/ULIP-2/pointnext/*.pt
```

If your download tool creates an extra nesting level, move or symlink the
downloaded `PCs` directory so it is available at `datasets/PCs`.

## Expected Data Layout

Both methods read feature tensors under `datasets/PCs`.

```text
datasets/PCs/<DATASET>/<ULIP-1-or-ULIP-2>/<BACKBONE>/*.pt
```

For example:

```text
datasets/PCs/Shapenet/ULIP-2/pointnext/*.pt
datasets/PCs/InstantMesh/ULIP-2/pointnext/*.pt
```

Feature filenames should contain a category and sample id separated by `-`, for
example:

```text
02691156-1a04e3eab45ca15dd86060f189eb133.pt
```

The split files `train_ids.txt`, `val_ids.txt`, and `test_ids.txt` must contain
one `category<TAB>sample_id` pair per line. The training scripts check that the
three splits are disjoint.

## Dataset Names

The fake dataset is one of:

- `InstantMesh`
- `LGM`
- `SAM3D`
- `TRELLIS`
- `TRELLIS_text`

Dataset abbreviations used in checkpoint names:

- `SI`: ShapeNet + InstantMesh
- `SL`: ShapeNet + LGM
- `SS`: ShapeNet + SAM3D
- `ST`: ShapeNet + TRELLIS
- `STT`: ShapeNet + TRELLIS_text

## Evaluation

`train_hsic.py` trains for 100 epochs in `networks/trainer.py`, saves a
checkpoint every epoch, then evaluates checkpoints on 10 fixed combinations:

- Objaverse vs InstantMesh
- Objaverse vs LGM
- Objaverse vs SAM3D
- Objaverse vs TRELLIS
- Objaverse vs TRELLIS_text
- ShapeNet vs InstantMesh
- ShapeNet vs LGM
- ShapeNet vs SAM3D
- ShapeNet vs TRELLIS
- ShapeNet vs TRELLIS_text

HSIC evaluation reports include `acc`, `ap`, `r_acc`, `f_acc`, `threshold`,
and `failures`.

`train_vib.py` evaluates the best Optuna trial on the held-out test split and
then evaluates that same best run on the same 10 fixed testing combinations.
It writes `summary.csv` for the same-pair test split and
`summary_10_combos.csv` for the 10-combo evaluation.

## Customizing

The wrappers are simple Bash loops. Edit the relevant wrapper to change common
settings:

- Change `--gpu_ids` to select a different GPU.
- For HSIC, add `--lambda_x_list` and `--lambda_y_list` to reduce or expand the
  lambda grid.
- For VIB-Net, add `--n_trials`, `--beta_min`, `--beta_max`, `--lr_min`, or
  `--lr_max` to change the Optuna search.
- Change `--name` to write into a different checkpoint namespace.
- Change `--checkpoints_dir` to write checkpoints somewhere else.

## Notes

- Both methods use CUDA when available and fall back to CPU if CUDA is
  unavailable or `--gpu_ids -1` is used.
- Both methods infer `--feature_dim` from the input `.pt` features and override
  the default when needed.
- `Objaverse` real datasets and `TRELLIS_text` fake datasets are automatically
  handled in unpaired sampling mode.
