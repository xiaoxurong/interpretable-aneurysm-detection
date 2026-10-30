"""Train PC-CBM with subject-grouped 5-fold cross-validation.

    python scripts/train.py --config configs/train_intra.yaml
    python scripts/train.py --config configs/train_intra.yaml --set model.backbone=DGCNN --folds 1 2

Writes to ``<save_dir>/<experiment.name>/``:
    best_model_fold_k.pth   checkpoint (+ config, concept names, chosen threshold)
    history_fold_k.csv      per-epoch train losses / val metrics
    oof_predictions.csv     out-of-fold test probabilities and concept scores
    summary.json            per-fold metrics and mean ± std over *all* folds
"""

from __future__ import annotations

import argparse
import os

from pointcbm.config import get_device, load_config
from pointcbm.engine import load_concept_bank, run_fold, save_run_summary
from pointcbm.models import build_model, count_parameters

KEYS = ["auroc", "auprc", "bal_acc", "acc0", "acc1", "f1", "ece"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    device = get_device(args.device)
    folds = args.folds or cfg["data"]["folds"]
    out_dir = os.path.join(cfg["experiment"]["save_dir"], cfg["experiment"]["name"])

    E_text, names = load_concept_bank(cfg, device)
    n_params = count_parameters(build_model(cfg, d_t=E_text.shape[1], num_concepts=E_text.shape[0]))
    print(
        f"[{cfg['experiment']['name']}] {cfg['model']['backbone']}/{cfg['model'].get('head', 'cbm')} "
        f"| {n_params / 1e6:.2f}M params | {len(names)} concepts | device={device}"
    )

    results = []
    for fold in folds:
        r = run_fold(cfg, fold, E_text, names, device, out_dir)
        m = r.test_metrics
        print(f"fold {fold} TEST | " + " ".join(f"{k} {m[k]:.3f}" for k in KEYS) + f" | thr {r.threshold:.3f}")
        results.append(r)

    summary = save_run_summary(results, cfg, out_dir)
    print("\nmean ± std over folds", folds)
    for k in KEYS:
        mu, sd = summary["mean_std"][k]
        print(f"  {k:8s} {mu:.3f} ± {sd:.3f}")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
