"""Evaluation metrics for imbalanced binary classification.

Includes the pieces that are easy to get wrong on a 9:1 imbalanced medical task:
operating-point selection on validation data (never on test), calibration error,
and a *subject-level* (cluster) bootstrap for confidence intervals — segments
from the same subject are correlated, so resampling segments independently
would understate the uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: Sequence[int], prob: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "acc": (tp + tn) / max(len(y), 1),
        "acc0": tn / max(tn + fp, 1),  # specificity (vessel)
        "acc1": tp / max(tp + fn, 1),  # sensitivity (aneurysm)
        "bal_acc": balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else float("nan"),
        "f1": f1_score(y, pred, zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision": tp / max(tp + fp, 1),
        "threshold": float(threshold),
    }
    if len(np.unique(y)) > 1:
        out["auroc"] = roc_auc_score(y, p)
        out["auprc"] = average_precision_score(y, p)
    else:
        out["auroc"] = out["auprc"] = float("nan")
    out["ece"] = expected_calibration_error(y, p)
    out["brier"] = float(np.mean((p - y) ** 2))
    return out


def select_threshold(y_true: Sequence[int], prob: Sequence[float], method: str = "youden") -> float:
    """Pick an operating point on *validation* predictions.

    ``youden`` maximises sensitivity + specificity - 1; ``f1`` maximises F1 of the
    positive class. Returns 0.5 if the validation set has a single class.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    if method == "youden":
        fpr, tpr, thr = roc_curve(y, p)
        j = tpr - fpr
        t = thr[int(np.argmax(j))]
        return float(np.clip(t, 1e-6, 1 - 1e-6))
    if method == "f1":
        cands = np.unique(p)
        scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in cands]
        return float(cands[int(np.argmax(scores))])
    if method == "fixed":
        return 0.5
    raise ValueError(method)


def expected_calibration_error(y_true: Sequence[int], prob: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    p = np.asarray(prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def cluster_bootstrap_ci(
    y_true: Sequence[int],
    prob: Sequence[float],
    groups: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) percentile CI, resampling *subjects* with replacement."""
    y = np.asarray(y_true)
    p = np.asarray(prob)
    g = np.asarray(groups)
    uniq = np.unique(g)
    members = {u: np.flatnonzero(g == u) for u in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, p)), float(lo), float(hi)


def is_collapsed(y_true: Sequence[int], prob: Sequence[float], threshold: float = 0.5) -> bool:
    """True if the classifier predicts (almost) only one class at this threshold."""
    pred = (np.asarray(prob) >= threshold).astype(int)
    frac_pos = pred.mean()
    return bool(frac_pos < 0.01 or frac_pos > 0.99)


def summarize_folds(fold_metrics: list[dict[str, float]], keys: Sequence[str] | None = None) -> dict[str, tuple[float, float]]:
    """Mean and (population) std across folds — over *all* folds, none excluded."""
    keys = keys or [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    out = {}
    for k in keys:
        v = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out
