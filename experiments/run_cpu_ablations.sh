#!/usr/bin/env bash
# Controlled ablations on the compact backbone (one factor changed per run).
# ~20 min per run on a 2-core CPU; all runs share seeds and folds.
set -euo pipefail
CFG=configs/cpu_ablation.yaml
run() { python scripts/train.py --config "$CFG" --set experiment.name="$1" "${@:2}"; }

run lite_cbm_clip                                             # PC-CBM, CLIP concept bank
run lite_blackbox      model.head=blackbox                    # same backbone, no bottleneck
run lite_cbm_random    data.concept_embeddings=random         # random "concepts" (language-grounding control)
run lite_cbm_no_supcon loss_weights.lambda_supcon=0.0         # focal loss only (no SupCon)
