"""Feature-based baselines with classical classifiers.

``--features geometric``  44 hand-crafted shape descriptors (pointcbm/geometry.py)
``--features openshape``  zero-shot OpenShape point-concept similarity maps
                          (data/OpenShape/heatmaps, 21 concepts) pooled to
                          per-concept max / mean / p90 -> "what does a frozen
                          general-purpose 3D-language model already know?"

Uses exactly the same subject-grouped folds and the same protocol as the deep
models (fit on train, pick threshold on val, score test once).

    python scripts/baseline_geometric.py --features geometric --out results/geometric_baseline
    python scripts/baseline_geometric.py --features openshape --out results/openshape_zeroshot
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pointcbm.data import fold_csv, subject_id
from pointcbm.geometry import segment_features
from pointcbm.metrics import binary_metrics, select_threshold, summarize_folds


def openshape_features(path: str) -> dict[str, float]:
    import torch

    d = torch.load(path, map_location="cpu", weights_only=False)
    h = np.asarray(d["heatmap"], dtype=np.float64)  # (N, 21) point-concept similarity
    out = {}
    for k, name in enumerate(d["concept_names"]):
        out[f"{name}|max"] = h[:, k].max()
        out[f"{name}|mean"] = h[:, k].mean()
        out[f"{name}|p90"] = np.percentile(h[:, k], 90)
    return out


MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000)),
    "gbdt": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_IntrA_6d/points/")
    ap.add_argument("--csv_dir", default="data/processed_IntrA_6d/")
    ap.add_argument("--out", default="results/geometric_baseline")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--features", choices=["geometric", "openshape"], default="geometric")
    ap.add_argument("--heatmap_dir", default="data/OpenShape/heatmaps")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cache = os.path.join(args.out, "features.csv")
    if os.path.exists(cache):
        feats = pd.read_csv(cache, index_col=0)
    else:
        files = pd.read_csv(os.path.join(args.csv_dir, "all_labels.csv")).filename.tolist()
        if args.features == "geometric":
            rows = Parallel(n_jobs=args.jobs)(delayed(segment_features)(np.load(os.path.join(args.data_dir, f))) for f in files)
        else:
            rows = [openshape_features(os.path.join(args.heatmap_dir, f.replace(".npy", "_heatmap.pt"))) for f in files]
        feats = pd.DataFrame(rows, index=files)
        feats.to_csv(cache)
    print(f"{feats.shape[1]} features x {len(feats)} segments")

    summary, oof = {}, []
    for name, make in MODELS.items():
        per_fold = []
        for fold in range(1, 6):
            sp = {s: pd.read_csv(fold_csv(args.csv_dir, fold, s)) for s in ("train", "val", "test")}
            X = {s: feats.loc[d.filename].to_numpy() for s, d in sp.items()}
            y = {s: d.label.to_numpy() for s, d in sp.items()}
            clf = make().fit(X["train"], y["train"])
            thr = select_threshold(y["val"], clf.predict_proba(X["val"])[:, 1])
            p = clf.predict_proba(X["test"])[:, 1]
            m = binary_metrics(y["test"], p, thr)
            per_fold.append(m)
            oof.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "filename": sp["test"].filename,
                        "subject": sp["test"].filename.map(subject_id),
                        "label": y["test"],
                        "prob": p,
                        "threshold": thr,
                    }
                )
            )
        ms = summarize_folds(per_fold, ["auroc", "auprc", "bal_acc", "acc0", "acc1", "f1", "ece"])
        summary[name] = {"per_fold": per_fold, "mean_std": ms}
        print(f"{name:7s} " + " ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in ms.items()))

        if name == "logreg":
            labels = pd.read_csv(os.path.join(args.csv_dir, "all_labels.csv")).set_index("filename").loc[feats.index].label
            coef = make().fit(feats.to_numpy(), labels)[-1].coef_[0]
            top = pd.Series(coef, index=feats.columns).sort_values(key=np.abs, ascending=False).head(10)
            summary[name]["top_coefficients"] = top.round(3).to_dict()
            print("  top standardized logreg coefficients:", ", ".join(f"{k}={v:+.2f}" for k, v in top.items()))

    pd.concat(oof).to_csv(os.path.join(args.out, "oof_predictions.csv"), index=False)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
