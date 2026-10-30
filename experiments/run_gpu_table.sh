#!/usr/bin/env bash
# Reproduce the main results table: 3 backbones x {focal, focal + SupCon}, 5 folds each.
set -euo pipefail
for bb in PointNet PointNetPlusPlus DGCNN; do
  for sc in 0.0 0.01; do
    tag=$([ "$sc" = "0.0" ] && echo focal || echo focal_supcon)
    python scripts/train.py --config configs/train_intra.yaml \
      --set model.backbone=$bb loss_weights.lambda_supcon=$sc experiment.name=IntrA_PCCBM_${bb}_${tag}
  done
done
