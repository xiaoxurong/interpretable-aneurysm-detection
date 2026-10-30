"""Aggregate experiment outputs into the tables and figures used in the README.

Reads ``oof_predictions.csv`` from each run, so nothing is re-trained:

    python scripts/analyze_results.py --out docs/figures

Produces
    results/results_table.md        per-run mean ± std over folds + pooled AUROC with subject-bootstrap 95% CI
    docs/figures/roc_pr.png         pooled out-of-fold ROC and precision-recall curves
    docs/figures/concepts.png       concept-score distributions by class + mean-ablation importance
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from pointcbm.metrics import binary_metrics, cluster_bootstrap_ci, summarize_folds

# Reference categorical palette (fixed order) + text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"

RUNS = [
    ("Geometric features + GBDT", "results/geometric_baseline", "gbdt"),
    ("Geometric features + LogReg", "results/geometric_baseline", "logreg"),
    ("OpenShape zero-shot concept maps + LogReg", "results/openshape_zeroshot", "logreg"),
    ("PC-CBM (CLIP concepts)", "results/lite_cbm_clip", None),
    ("Black-box head (no bottleneck)", "results/lite_blackbox", None),
    ("PC-CBM (random concepts)", "results/lite_cbm_random", None),
    ("PC-CBM (no SupCon)", "results/lite_cbm_no_supcon", None),
]
ACTIVE = 0.5  # concept counted as "firing" above this pooled score
KEYS = ["auroc", "auprc", "bal_acc", "acc0", "acc1", "f1_macro", "ece"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def load_oof(path: str, model: str | None) -> pd.DataFrame | None:
    f = os.path.join(path, "oof_predictions.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    return df[df.model == model] if model else df


def results_table(runs) -> str:
    lines = [
        "| Model | AUROC (pooled, 95% CI) | AUROC (fold mean ± std) | AUPRC | Bal. acc | Spec. (ACC₀) | Sens. (ACC₁) | Macro-F1 | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, df in runs:
        per_fold = [binary_metrics(g.label, g.prob, g.threshold.iloc[0]) for _, g in df.groupby("fold")]
        ms = summarize_folds(per_fold, KEYS)
        auc, lo, hi = cluster_bootstrap_ci(df.label, df.prob, df.subject, roc_auc_score, n_boot=2000)
        cells = [f"{ms[k][0]:.3f} ± {ms[k][1]:.3f}" for k in KEYS]
        lines.append(f"| {name} | **{auc:.3f}** ({lo:.3f}–{hi:.3f}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_roc_pr(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    prevalence = runs[0][1].label.mean()
    for (name, df), color in zip(runs, SERIES):
        fpr, tpr, _ = roc_curve(df.label, df.prob)
        prec, rec, _ = precision_recall_curve(df.label, df.prob)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name}  ({roc_auc_score(df.label, df.prob):.3f})")
        axes[1].plot(rec, prec, color=color, lw=2, label=f"{name}  ({average_precision_score(df.label, df.prob):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    axes[1].axhline(prevalence, ls="--", color=MUTED, lw=1)
    axes[1].text(0.01, prevalence + 0.02, f"prevalence = {prevalence:.2f} (chance AUPRC)", ha="left", color=INK2, fontsize=8)
    for ax, (xl, yl, t) in zip(
        axes,
        [
            ("False positive rate", "True positive rate", "ROC  (AUROC)"),
            ("Recall (sensitivity)", "Precision", "Precision–recall  (AUPRC)"),
        ],
    ):
        style(ax)
        ax.set_xlabel(xl, color=INK2)
        ax.set_ylabel(yl, color=INK2)
        ax.set_title(t, color=INK, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right" if ax is axes[0] else "upper right")
    fig.suptitle(
        "Pooled out-of-fold predictions, 1,909 vessel segments / 103 subjects", color=INK2, fontsize=9, x=0.01, ha="left"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def concept_analysis(run_dir: str, out: str) -> dict:
    """Concept-score distributions by class and mean-ablation importance (ΔAUROC)."""
    df = pd.read_csv(os.path.join(run_dir, "oof_predictions.csv"))
    ccols = sorted([c for c in df.columns if c.startswith("c") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    ck = torch.load(os.path.join(run_dir, "best_model_fold_1.pth"), map_location="cpu", weights_only=False)
    names = ck["concept_names"]

    base = roc_auc_score(df.label, df.prob)
    drops, weights = [], []
    for k in range(len(ccols)):
        probs = []
        for fold, g in df.groupby("fold"):
            sd = torch.load(os.path.join(run_dir, f"best_model_fold_{fold}.pth"), map_location="cpu", weights_only=False)[
                "model_state_dict"
            ]
            W, b = sd["predictor.weight"].numpy(), sd["predictor.bias"].numpy()
            C = g[ccols].to_numpy().copy()
            C[:, k] = np.median(C[:, k])  # replace concept k by an uninformative constant
            logits = C @ W.T + b
            probs.append(pd.Series(np.exp(logits[:, 1]) / np.exp(logits).sum(1), index=g.index))
            if k == 0:
                weights.append(W[1] - W[0])
        drops.append(base - roc_auc_score(df.label, pd.concat(probs).loc[df.index]))
    w = np.mean(weights, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor=SURFACE)
    short = [textwrap.fill(n.replace("a ", "", 1), 34) for n in names]
    pos = np.arange(len(ccols))
    # Concept scores are strongly bimodal (off ≈ 0.39, on > 1), so show how often each concept fires.
    for j, (lab, color, txt) in enumerate([(0, SERIES[0], "vessel segments"), (1, SERIES[1], "aneurysm segments")]):
        frac = [(df.loc[df.label == lab, c] > ACTIVE).mean() for c in ccols]
        axes[0].barh(pos + (0.5 - j) * 0.38, frac, height=0.36, color=color, edgecolor=SURFACE, linewidth=2, label=txt)
        for y, f in zip(pos + (0.5 - j) * 0.38, frac):
            axes[0].text(f + 0.01, y, f"{f:.0%}", va="center", fontsize=7.5, color=INK2)
    axes[0].set_yticks(pos, short, color=INK, fontsize=8)
    axes[0].set_xlim(0, 1.08)
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].set_xlabel(f"share of held-out segments where the concept fires (Ĉ > {ACTIVE})", color=INK2)
    axes[0].set_title("Concept activation by class", color=INK, loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    order = np.argsort(drops)
    axes[1].barh(pos, np.array(drops)[order], color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
    axes[1].axvline(0, color=MUTED, lw=0.8)
    axes[1].set_yticks(pos, [short[i] for i in order], color=INK, fontsize=8)
    for y, i in enumerate(order):
        axes[1].text(max(drops[i], 0) + 0.001, y, f"{drops[i]:+.3f}   w̄ = {w[i]:+.2f}", va="center", fontsize=7.5, color=INK2)
    axes[1].set_xlabel("pooled AUROC drop when the concept is replaced by a constant", color=INK2)
    axes[1].set_title("Concept ablation", color=INK, loc="left", fontsize=11)
    axes[1].set_xlim(min(0, min(drops)) - 0.003, max(drops) * 1.9 + 0.005)
    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return {
        "base_auroc": base,
        "concepts": {n: {"auroc_drop": float(d), "aneurysm_weight": float(wi)} for n, d, wi in zip(names, drops, w)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [(n, d) for n, p, m in RUNS if (d := load_oof(p, m)) is not None]
    table = results_table(runs)
    with open("results/results_table.md", "w") as f:
        f.write(table + "\n")
    print(table)

    show = [
        "Geometric features + GBDT",
        "OpenShape zero-shot concept maps + LogReg",
        "PC-CBM (CLIP concepts)",
        "Black-box head (no bottleneck)",
        "PC-CBM (random concepts)",
    ]
    curve_runs = [r for r in runs if r[0] in show]
    plot_roc_pr(curve_runs, os.path.join(args.out, "roc_pr.png"))
    if os.path.exists("results/lite_cbm_clip/best_model_fold_1.pth"):
        info = concept_analysis("results/lite_cbm_clip", os.path.join(args.out, "concepts.png"))
        with open("results/concept_importance.json", "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
