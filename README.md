# Syn3D-Bench

This repository trains feature-based synthetic-3D detectors with two training
entry points:

- `train_hsic.py`: HSIC Bottleneck detector
- `train_vib.py`: VIB-Net detector

Both methods train ShapeNet as the real dataset against one generated fake
dataset, using precomputed point-cloud feature tensors.

## Running From Repository Root

Run commands from the repository root because the training scripts use relative
paths such as `datasets/PCs`, `train_ids.txt`, `val_ids.txt`, and
`test_ids.txt`.

```bash
cd /path/to/Syn3D-Bench
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

- `--n_trials 100`
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

Download the feature archives into the repository root with the Hugging Face
Hub CLI. The remote paths already start with `datasets/`, so using the
repository root as `--local-dir` preserves the expected `datasets/PCs` layout.

```bash
cd /path/to/Syn3D-Bench
huggingface-cli download jamesyoung0623/Syn3D-Dataset \
  --repo-type dataset \
  --include "datasets/PCs/**" \
  --local-dir .
```

The feature folders are stored as split `.tar.gz` archives. Extract each
backbone's parts before training:

```bash
cd /path/to/Syn3D-Bench
find datasets/PCs -name '*.part*.tar.gz' -print0 | while IFS= read -r -d '' shard; do
  tar -xzf "$shard" -C "$(dirname "$shard")"
done
```

After extraction, confirm the point-cloud feature tree contains `.pt` tensors
at paths such as:

```text
datasets/PCs/Shapenet/ULIP-2/pointnext/*.pt
datasets/PCs/InstantMesh/ULIP-2/pointnext/*.pt
```

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

## Vision-Language Inference

The inference scripts run VLM classification over rendered videos from all six
project folders:

- `ULIP`
- `InstantMesh`
- `LGM`
- `SAM3D`
- `TRELLIS`
- `TRELLIS_text`

Shared project iteration, output paths, and optional logging live in
`inference_common.py`. By default, it looks for project folders under
`/path/to/project/root`. Override this root with:

```bash
export SYN3D_PROJECT_ROOT=/path/to/project/root
```

The default video folders are:

```text
ULIP/ulip/ULIP_Shapenet_Triplets/videos_white
InstantMesh/outputs/videos_white
LGM/outputs/videos_white
SAM3D/outputs/videos_black
TRELLIS/outputs/videos_black
TRELLIS_text/outputs_text/videos_black
```

Each model script loops over all six projects and writes one result JSON into
each project directory:

```bash
cd /path/to/Syn3D-Bench
python inference_Idefics2.py
python inference_InternVL2.py
python inference_InternVL3.py
python inference_LLAVA.py
python inference_LongVA.py
python inference_Mantis.py
python inference_mPLUG-Owl3.py
python inference_Phi.py
python inference_Qwen2-VL.py
python inference_VILA.py
```

Common helper files:

- `inference_common.py`: dataset/project iteration, video path resolution, and
  log tee setup.
- `inference_phi35_common.py`: shared Phi-3.5-Vision inference implementation
  used by `inference_Phi.py`.

Example outputs:

```text
/path/to/project/root/ULIP/llava_ov_7b_all_results.json
/path/to/project/root/InstantMesh/llava_ov_7b_all_results.json
/path/to/project/root/SAM3D/llava_ov_7b_all_results.json
```

### InternVL3 Fine-Tuning And Inference

`run_internvl3_finetuned.sh` combines InternVL3 SFT training and inference for
the five fine-tuning combos:

```text
SI SL SS ST STT
```

Usage:

```bash
cd /path/to/Syn3D-Bench
./run_internvl3_finetuned.sh train
./run_internvl3_finetuned.sh infer
./run_internvl3_finetuned.sh all
```

`all` is the default mode. The training entrypoint is
`train_internvl3_sft.py` in this repository. The script expects fine-tuning
JSONL files and checkpoints under:

```text
internvl3_finetune/
```

Override paths and runtime settings with environment variables:

```bash
ROOT_DIR=/path/to/root \
INTERNVL3_FINETUNE_ROOT=/path/to/internvl3_finetune \
INTERNVL3_MODEL_NAME=OpenGVLab/InternVL3_5-4B-Instruct \
INTERNVL3_MODEL_SIZE=4B \
INTERNVL3_RUN_TAG=internvl3_5_4b \
INTERNVL3_CUDA_VISIBLE_DEVICES=0,1 \
INTERNVL3_NPROC_PER_NODE=2 \
./run_internvl3_finetuned.sh all
```

`INTERNVL3_MODEL_NAME`, `INTERNVL3_MODEL_SIZE`, and `INTERNVL3_RUN_TAG` are
required. Checkpoints are written under
`internvl3_finetune/checkpoints/<INTERNVL3_RUN_TAG>_<COMBO>_sft/`. For the 4B
InternVL3.5 checkpoint:

```bash
INTERNVL3_MODEL_NAME=OpenGVLab/InternVL3_5-4B-Instruct \
INTERNVL3_MODEL_SIZE=4B \
INTERNVL3_RUN_TAG=internvl3_5_4b \
./run_internvl3_finetuned.sh all
```

InternVL3 inference also supports model/runtime overrides, for example:

```bash
INTERNVL3_MODEL_NAME=OpenGVLab/InternVL3-2B \
INTERNVL3_FINETUNE_DIR=/path/to/checkpoint-final \
python inference_InternVL3.py
```

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
