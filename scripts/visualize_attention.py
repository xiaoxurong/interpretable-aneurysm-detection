"""Render where each concept "looks" on held-out vessel segments.

    python scripts/visualize_attention.py --run results/lite_cbm_clip --fold 1 --out docs/figures

Saves
    attention_gallery.png     correctly-flagged aneurysms + a vessel, coloured by the most aneurysm-indicative concept
    attention_all_concepts.png one aneurysm, one panel per concept
    and (optionally, --ply) coloured .ply files for MeshLab / Open3D.
"""

from __future__ import annotations

import argparse
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from pointcbm.config import deep_update, load_config
from pointcbm.data import IntrAClassificationDataset, fold_csv
from pointcbm.engine import load_concept_bank
from pointcbm.models import build_model
from pointcbm.visualization import export_concept_attention

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# Sequential, single hue: light (low attention) -> dark blue (high attention)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e4e3df", "#a9c6ec", "#2a78d6", "#0d3a75"])


def pca_align(xyz: np.ndarray) -> np.ndarray:
    c = xyz - xyz.mean(0)
    _, vecs = np.linalg.eigh(np.cov(c.T))
    return c @ vecs[:, ::-1]


def draw(ax, xyz, att, title, sub):
    p = pca_align(xyz)
    a = att  # absolute sigmoid attention in [0, 1] — comparable across panels
    order = np.argsort(a)  # draw high-attention points last (on top)
    ax.scatter(p[order, 0], p[order, 1], p[order, 2], c=a[order], cmap=SEQ, s=2.5, linewidths=0, vmin=0, vmax=1)
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect(np.ptp(p, axis=0))
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=9, pad=0)
    ax.text2D(0.5, -0.02, sub, transform=ax.transAxes, ha="center", color=INK2, fontsize=7.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/lite_cbm_clip")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--ply", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(os.path.join(args.run, f"best_model_fold_{args.fold}.pth"), map_location="cpu", weights_only=False)
    cfg = deep_update(load_config(None), ck["config"])
    E, names = load_concept_bank(cfg, torch.device("cpu"))
    model = build_model(cfg, d_t=E.shape[1], num_concepts=E.shape[0])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    thr = ck.get("threshold", 0.5)

    ds = IntrAClassificationDataset(fold_csv(cfg["data"]["csv_dir"], args.fold, "test"), cfg["data"]["data_dir"])
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = ds.labels
    with torch.no_grad():
        logits, C, A = [], [], []
        for xb in X.split(64):
            lo, c, a = model(xb, E)
            logits.append(lo), C.append(c), A.append(a)
        prob = torch.softmax(torch.cat(logits), 1)[:, 1].numpy()
        A = torch.cat(A).numpy()

    w = (model.predictor.weight[1] - model.predictor.weight[0]).detach().numpy()
    k_star = int(np.argmax(w))  # concept that pushes hardest toward "aneurysm"

    tp = [i for i in np.argsort(-prob) if y[i] == 1 and prob[i] >= thr][:3]
    tn = [i for i in np.argsort(prob) if y[i] == 0][:1]
    picks = tp + tn
    fig = plt.figure(figsize=(3.2 * len(picks), 3.6), facecolor=SURFACE)
    for j, i in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), j + 1, projection="3d")
        lab = "aneurysm" if y[i] else "vessel"
        draw(
            ax,
            X[i, :, :3].numpy(),
            A[i, :, k_star],
            f"{lab} — {ds.filenames[i].split('_', 1)[1][:-4]}",
            f"P(aneurysm) = {prob[i]:.2f}",
        )
    fig.suptitle(f"Point attention for concept: “{names[k_star]}”   (colour = sigmoid attention, 0 light → 1 dark)", color=INK, fontsize=10, y=0.98)
    fig.savefig(os.path.join(args.out, "attention_gallery.png"), dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    i = tp[0]
    fig = plt.figure(figsize=(3.0 * len(names), 3.5), facecolor=SURFACE)
    for k, name in enumerate(names):
        ax = fig.add_subplot(1, len(names), k + 1, projection="3d")
        draw(ax, X[i, :, :3].numpy(), A[i, :, k], textwrap.fill(name, 28), f"w_aneurysm = {w[k]:+.2f}")
    fig.suptitle(f"One held-out aneurysm segment, all {len(names)} concepts", color=INK, fontsize=10, y=1.0)
    fig.savefig(os.path.join(args.out, "attention_all_concepts.png"), dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    if args.ply:
        for k in range(len(names)):
            export_concept_attention(X[i, :, :3], A[i, :, k], os.path.join(args.out, f"aneurysm_concept{k}.ply"))
    print(f"saved figures to {args.out} (concept used for gallery: '{names[k_star]}')")


if __name__ == "__main__":
    main()
