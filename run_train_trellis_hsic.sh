#!/usr/bin/env bash
set -euo pipefail

# ULIP-2 supports: pointnext, pointbert_xyz
for pc_backbone in pointnext pointbert_xyz; do
  python train_hsic.py \
    --train_real_datasets Shapenet \
    --train_fake_datasets TRELLIS \
    --pc_variant ULIP-2 \
    --pc_backbone "${pc_backbone}" \
    --gpu_ids 0
done

# ULIP-1 supports: pointbert, pointnext, pointmlp, pointnet2_ssg
for pc_backbone in pointbert pointnext pointmlp pointnet2_ssg; do
  python train_hsic.py \
    --train_real_datasets Shapenet \
    --train_fake_datasets TRELLIS \
    --pc_variant ULIP-1 \
    --pc_backbone "${pc_backbone}" \
    --gpu_ids 0
done
