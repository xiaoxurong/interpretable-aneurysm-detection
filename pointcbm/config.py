"""YAML config loading with dotted command-line overrides.

    python scripts/train.py --config configs/train_intra.yaml \
        --set model.backbone=DGCNN training.num_epochs=50 loss_weights.lambda_supcon=0
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {"name": "pointcbm", "task": "classification", "save_dir": "checkpoints/", "seed": 42},
    "data": {
        "data_dir": "data/processed_IntrA_6d/points/",
        "csv_dir": "data/processed_IntrA_6d/",
        "concept_embeddings": "data/metadata/concept_embeds.pt",
        "num_points": None,
        "rotate": False,
        "folds": [1, 2, 3, 4, 5],
    },
    "model": {
        "backbone": "PointNet",
        "head": "cbm",
        "d_f": 128,
        "d_hidden": 256,
        "in_channel": 6,
        "num_classes": 2,
        "k": 20,
        "pooling": "topk_lse",
    },
    "training": {
        "num_epochs": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 15,
        "warmup_epochs": 30,
        "monitor": "auroc",
        "scheduler": "plateau",
        "grad_clip": 1.0,
        "num_workers": 0,
        "amp": False,
        "balanced_sampler": True,
        "threshold_method": "youden",
    },
    "loss_weights": {
        "alpha": 0.5,
        "gamma": 2.0,
        "lambda_align": 0.0,
        "lambda_sparse": 0.01,
        "lambda_ortho": 0.01,
        "lambda_supcon": 0.01,
        "temperature": 0.1,
        "lambda_l1": 1e-4,
    },
    "logging": {"wandb": False, "project": "IntrA-CBM-CV"},
}


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.subkey=value")
        k, v = item.split("=", 1)
        set_by_path(cfg, k, yaml.safe_load(v))
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
