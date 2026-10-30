"""Training / evaluation loops shared by every script.

One call to :func:`run_fold` = train on ``fold_k/train``, early-stop on
``fold_k/val``, choose the decision threshold on ``fold_k/val``, then score
``fold_k/test`` exactly once. Out-of-fold test predictions are written to CSV so
all downstream analysis (bootstrap CIs, calibration, plots) is reproducible
without re-running inference.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import seed_everything
from .data import IntrAClassificationDataset, fold_csv, make_balanced_sampler, subject_id
from .losses import CBMLoss
from .metrics import binary_metrics, is_collapsed, select_threshold
from .models import ConceptBottleneck3D, build_model, random_concept_bank

# ── concept bank ─────────────────────────────────────────────────────────────

DEFAULT_BANK = "data/metadata/concept_embeds.pt"


def load_concept_bank(cfg: dict, device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """Load frozen CLIP text embeddings, or a shape-matched random control bank."""
    spec = cfg["data"]["concept_embeddings"]
    if isinstance(spec, str) and spec.startswith("random"):
        # "random" or "random:<path-to-real-bank>" -> same (K, d_t) as the real bank
        ref = spec.split(":", 1)[1] if ":" in spec else DEFAULT_BANK
        real = torch.load(ref, map_location="cpu")["embeddings"]
        E = random_concept_bank(real.shape[0], real.shape[1], seed=cfg["experiment"].get("seed", 0))
        names = [f"random_{i}" for i in range(real.shape[0])]
        return E.to(device).float(), names
    data = torch.load(spec, map_location="cpu")
    names = list(data.get("concept_names", [f"concept_{i}" for i in range(len(data["embeddings"]))]))
    return data["embeddings"].to(device).float(), names


# ── loops ────────────────────────────────────────────────────────────────────


def _worker_init(worker_id: int) -> None:
    # Give each DataLoader worker its own augmentation RNG (otherwise forked
    # workers share an identical numpy Generator state and repeat augmentations).
    info = torch.utils.data.get_worker_info()
    info.dataset._rng = np.random.default_rng(torch.initial_seed() % 2**32)


def train_one_epoch(model, loader, criterion, optimizer, E_text, device, grad_clip=1.0, scaler=None):
    model.train()
    totals: dict[str, float] = {}
    n = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits, C_hat, A = model(X, E_text)
            pw = model.predictor.weight if isinstance(model, ConceptBottleneck3D) else None
            out = criterion(logits.float(), y, C_hat=C_hat, A=A, predictor_weight=pw)
        if scaler is not None:
            scaler.scale(out.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out.total.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        for k, v in out.as_floats().items():
            totals[k] = totals.get(k, 0.0) + v * len(y)
        n += len(y)
    return {f"loss_{k}": v / n for k, v in totals.items()}


@torch.no_grad()
def predict(model, loader, E_text, device, return_concepts: bool = False):
    """Return P(aneurysm), labels and (optionally) concept scores for a loader."""
    model.eval()
    probs, labels, concepts = [], [], []
    for X, y in loader:
        logits, C_hat, _ = model(X.to(device), E_text)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
        if return_concepts and C_hat is not None:
            concepts.append(C_hat.cpu())
    out = {"prob": torch.cat(probs).numpy(), "label": torch.cat(labels).numpy()}
    if return_concepts and concepts:
        out["concepts"] = torch.cat(concepts).numpy()
    return out


# ── one fold ─────────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    fold: int
    val_metrics: dict
    test_metrics: dict
    threshold: float
    best_epoch: int
    epochs_run: int
    train_seconds: float
    checkpoint: str
    predictions: pd.DataFrame = field(repr=False)


def _make_loaders(cfg, fold):
    d, t = cfg["data"], cfg["training"]
    common = dict(data_dir=d["data_dir"], num_points=d.get("num_points"))
    train_ds = IntrAClassificationDataset(
        fold_csv(d["csv_dir"], fold, "train"), augment=True, rotate=d.get("rotate", False), **common
    )
    val_ds = IntrAClassificationDataset(fold_csv(d["csv_dir"], fold, "val"), **common)
    test_ds = IntrAClassificationDataset(fold_csv(d["csv_dir"], fold, "test"), **common)

    nw = t.get("num_workers", 0)
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available(), worker_init_fn=_worker_init if nw else None)
    seed = cfg["experiment"].get("seed", 42) + fold
    if t.get("balanced_sampler", True):
        train_loader = DataLoader(
            train_ds, batch_size=t["batch_size"], sampler=make_balanced_sampler(train_ds.labels, seed), drop_last=True, **kw
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True, **kw)
    eval_bs = t.get("eval_batch_size", t["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **kw)
    return train_loader, val_loader, test_loader, test_ds, val_ds


def run_fold(
    cfg: dict, fold: int, E_text: torch.Tensor, concept_names: list[str], device: torch.device, out_dir: str, log=print
) -> FoldResult:
    t_cfg = cfg["training"]
    seed_everything(cfg["experiment"].get("seed", 42) + fold)
    train_loader, val_loader, test_loader, test_ds, val_ds = _make_loaders(cfg, fold)

    model = build_model(cfg, d_t=E_text.shape[1], num_concepts=E_text.shape[0]).to(device)
    criterion = CBMLoss.from_config(cfg["loss_weights"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"])
    num_epochs = t_cfg["num_epochs"]
    if t_cfg.get("scheduler", "plateau") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda") if (t_cfg.get("amp") and device.type == "cuda") else None

    wandb_run = None
    if cfg.get("logging", {}).get("wandb"):
        import wandb

        wandb_run = wandb.init(
            project=cfg["logging"]["project"],
            group=cfg["experiment"]["name"],
            name=f"{cfg['experiment']['name']}_fold{fold}",
            config=cfg,
            reinit=True,
        )

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"best_model_fold_{fold}.pth")
    monitor = t_cfg.get("monitor", "auroc")
    patience, warmup = t_cfg.get("patience", 15), t_cfg.get("warmup_epochs", 0)
    best, best_epoch, bad_epochs, history = -np.inf, 0, 0, []
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, E_text, device, t_cfg.get("grad_clip", 1.0), scaler
        )
        val_pred = predict(model, val_loader, E_text, device)
        val_m = binary_metrics(val_pred["label"], val_pred["prob"])
        score = val_m[monitor]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(score)
        else:
            scheduler.step()

        row = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_m.items()}, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if wandb_run is not None:
            wandb_run.log(row)

        improved = score > best
        if improved:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save(
                {
                    "epoch": epoch,
                    "fold": fold,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "concept_names": concept_names,
                    f"best_val_{monitor}": best,
                },
                ckpt_path,
            )
        elif epoch > warmup:
            # Patience only starts counting once the warm-up period is over.
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            log(
                f"  fold {fold} | ep {epoch:3d} | loss {train_stats['loss_total']:.4f} | "
                f"val auroc {val_m['auroc']:.3f} auprc {val_m['auprc']:.3f}{'  *' if improved else ''}"
            )
        if bad_epochs >= patience:
            log(f"  fold {fold} | early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    train_seconds = time.time() - t0
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])

    # Operating point chosen on validation only, then applied to test.
    val_pred = predict(model, val_loader, E_text, device)
    thr = select_threshold(val_pred["label"], val_pred["prob"], t_cfg.get("threshold_method", "youden"))
    test_pred = predict(model, test_loader, E_text, device, return_concepts=True)
    test_m = binary_metrics(test_pred["label"], test_pred["prob"], threshold=thr)
    test_m_05 = binary_metrics(test_pred["label"], test_pred["prob"], threshold=0.5)
    test_m.update({f"{k}@0.5": test_m_05[k] for k in ("acc0", "acc1", "f1", "bal_acc")})
    test_m["collapsed@0.5"] = is_collapsed(test_pred["label"], test_pred["prob"], 0.5)

    preds = pd.DataFrame(
        {
            "fold": fold,
            "filename": test_ds.filenames,
            "subject": [subject_id(f) for f in test_ds.filenames],
            "label": test_pred["label"],
            "prob": test_pred["prob"],
            "threshold": thr,
        }
    )
    if "concepts" in test_pred:
        for i in range(len(concept_names)):
            preds[f"c{i}"] = test_pred["concepts"][:, i]
    pd.DataFrame(history).to_csv(os.path.join(out_dir, f"history_fold_{fold}.csv"), index=False)

    ck = torch.load(ckpt_path, map_location="cpu")
    ck["threshold"] = thr
    torch.save(ck, ckpt_path)

    if wandb_run is not None:
        wandb_run.summary.update({f"test_{k}": v for k, v in test_m.items()})
        wandb_run.finish()

    return FoldResult(
        fold,
        binary_metrics(val_pred["label"], val_pred["prob"], thr),
        test_m,
        thr,
        best_epoch,
        len(history),
        train_seconds,
        ckpt_path,
        preds,
    )


def save_run_summary(results: list[FoldResult], cfg: dict, out_dir: str) -> dict:
    from .metrics import summarize_folds

    preds = pd.concat([r.predictions for r in results], ignore_index=True)
    preds.to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    per_fold = [
        {
            "fold": r.fold,
            "threshold": r.threshold,
            "best_epoch": r.best_epoch,
            "epochs_run": r.epochs_run,
            "train_seconds": r.train_seconds,
            **r.test_metrics,
        }
        for r in results
    ]
    numeric = [{k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)} for m in per_fold]
    summary = {"experiment": cfg["experiment"]["name"], "config": cfg, "per_fold": per_fold, "mean_std": summarize_folds(numeric)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
