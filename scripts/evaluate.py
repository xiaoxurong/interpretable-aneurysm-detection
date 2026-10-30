"""Evaluate saved fold checkpoints on their held-out test folds.

    python scripts/evaluate.py --config configs/evaluate_intra.yaml
    python scripts/evaluate.py --checkpoint_dir checkpoints/my_run      # config read from checkpoint

All folds are reported and aggregated. Folds where the model predicts a single
class at the 0.5 threshold are *flagged* (``collapsed@0.5``) but never dropped:
excluding them would silently inflate the mean.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pointcbm.config import deep_update, get_device, load_config
from pointcbm.data import IntrAClassificationDataset, fold_csv, subject_id
from pointcbm.engine import load_concept_bank, predict
from pointcbm.metrics import binary_metrics, cluster_bootstrap_ci, is_collapsed, select_threshold, summarize_folds
from pointcbm.models import build_model

KEYS = ["auroc", "auprc", "bal_acc", "acc0", "acc1", "f1", "ece"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ckpt_dir = args.checkpoint_dir or cfg.get("evaluation", {}).get("checkpoint_dir")
    if ckpt_dir is None:
        raise SystemExit("Pass --checkpoint_dir or set evaluation.checkpoint_dir in the config")
    device = get_device(args.device)

    paths = sorted(glob.glob(os.path.join(ckpt_dir, "best_model_fold_*.pth")))
    if not paths:
        raise SystemExit(f"No best_model_fold_*.pth in {ckpt_dir}")

    rows, frames = [], []
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        fold = int(ck.get("fold") or os.path.basename(path).split("_")[-1].split(".")[0])
        fcfg = deep_update(load_config(None), ck["config"]) if "config" in ck and args.config is None else cfg

        E_text, _ = load_concept_bank(fcfg, device)
        model = build_model(fcfg, d_t=E_text.shape[1], num_concepts=E_text.shape[0]).to(device)
        model.load_state_dict(ck.get("model_state_dict", ck), strict=True)

        def loader(split, fcfg=fcfg, fold=fold):
            ds = IntrAClassificationDataset(
                fold_csv(fcfg["data"]["csv_dir"], fold, split),
                fcfg["data"]["data_dir"],
                num_points=fcfg["data"].get("num_points"),
            )
            return ds, DataLoader(ds, batch_size=64, shuffle=False)

        if "threshold" in ck:
            thr = ck["threshold"]
        else:  # legacy checkpoints: pick the operating point on this fold's validation split
            _, vl = loader("val")
            v = predict(model, vl, E_text, device)
            thr = select_threshold(v["label"], v["prob"])
        test_ds, tl = loader("test")
        p = predict(model, tl, E_text, device)
        m = binary_metrics(p["label"], p["prob"], thr)
        m["collapsed@0.5"] = is_collapsed(p["label"], p["prob"])
        rows.append({"fold": fold, **m})
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "filename": test_ds.filenames,
                    "label": p["label"],
                    "prob": p["prob"],
                    "subject": [subject_id(f) for f in test_ds.filenames],
                }
            )
        )
        flag = "  [collapsed at 0.5 — kept in aggregate]" if m["collapsed@0.5"] else ""
        print(f"fold {fold} | " + " ".join(f"{k} {m[k]:.3f}" for k in KEYS) + f" | thr {thr:.3f}{flag}")

    ms = summarize_folds([{k: r[k] for k in KEYS} for r in rows])
    print(f"\nmean ± std over {len(rows)} folds (none excluded)")
    for k in KEYS:
        print(f"  {k:8s} {ms[k][0]:.3f} ± {ms[k][1]:.3f}")

    oof = pd.concat(frames)
    auc, lo, hi = cluster_bootstrap_ci(oof.label, oof.prob, oof.subject, roc_auc_score)
    print(f"\npooled out-of-fold AUROC {auc:.3f}  (95% subject-bootstrap CI {lo:.3f}–{hi:.3f}, n={len(oof)})")
    oof.to_csv(os.path.join(ckpt_dir, "eval_oof_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
