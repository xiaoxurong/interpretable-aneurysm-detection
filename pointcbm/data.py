"""IntrA vessel-segment dataset and subject-grouped cross-validation splits.

Each ``.npy`` file is one vessel segment: 2048 points x (xyz, unit normals).
File names encode the subject, e.g. ``aneurysm_ArteryObjAN117-2.npy`` is segment
2 of subject ``AN117``. All splits in ``data/processed_IntrA_6d/fold_k`` are
grouped by subject, so no subject contributes segments to both train and test.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

_SUBJECT_RE = re.compile(r"ArteryObj(AN\d+)")


def subject_id(filename: str) -> str:
    m = _SUBJECT_RE.search(filename)
    if m is None:
        raise ValueError(f"Cannot parse subject id from '{filename}'")
    return m.group(1)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random 3D rotation (QR of a Gaussian matrix with sign fix)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class IntrAClassificationDataset(Dataset):
    """Binary vessel (0) vs aneurysm (1) classification on IntrA segments.

    Args:
        csv_file: CSV with columns ``filename,label``.
        data_dir: directory holding the ``.npy`` point clouds.
        augment: random translation (±0.2) and isotropic scaling (0.67–1.5).
        rotate: additionally apply a random SO(3) rotation to xyz *and* normals.
        num_points: optionally subsample to this many points (random when
            augmenting, a fixed per-sample subset otherwise so eval is deterministic).
        cache: keep loaded arrays in memory (the full dataset is ~100 MB).
    """

    def __init__(
        self,
        csv_file: str,
        data_dir: str,
        augment: bool = False,
        rotate: bool = False,
        num_points: int | None = None,
        cache: bool = True,
    ):
        self.metadata = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.augment = augment
        self.rotate = rotate
        self.num_points = num_points
        self.cache = cache
        self._cache: dict[int, np.ndarray] = {}
        self._rng = np.random.default_rng()

    @property
    def labels(self) -> np.ndarray:
        return self.metadata.iloc[:, 1].to_numpy(dtype=np.int64)

    @property
    def filenames(self) -> list[str]:
        return self.metadata.iloc[:, 0].tolist()

    def __len__(self) -> int:
        return len(self.metadata)

    def _load(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]
        arr = np.load(os.path.join(self.data_dir, self.metadata.iloc[idx, 0])).astype(np.float32)
        if self.cache:
            self._cache[idx] = arr
        return arr

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        xyz, normals = pts[:, :3], pts[:, 3:]
        if self.rotate:
            R = random_rotation_matrix(self._rng).astype(np.float32)
            xyz, normals = xyz @ R.T, normals @ R.T
        xyz = xyz * self._rng.uniform(0.67, 1.5) + self._rng.uniform(-0.2, 0.2, size=(1, 3))
        return np.concatenate([xyz, normals], axis=1).astype(np.float32)

    def __getitem__(self, idx: int):
        pts = self._load(idx)
        if self.num_points is not None and self.num_points < len(pts):
            rng = self._rng if self.augment else np.random.default_rng(idx)
            pts = pts[rng.choice(len(pts), self.num_points, replace=False)]
        if self.augment:
            pts = self._augment(pts)
        return torch.from_numpy(np.ascontiguousarray(pts)), torch.tensor(self.labels[idx], dtype=torch.long)


def fold_csv(csv_dir: str, fold: int, split: str) -> str:
    return os.path.join(csv_dir, f"fold_{fold}", f"{split}_labels.csv")


def make_balanced_sampler(labels: np.ndarray, seed: int | None = None) -> WeightedRandomSampler:
    """Oversample the minority class so each batch is ~50/50 in expectation.

    Uses the label column directly (the original version loaded every point
    cloud from disk just to read its label).
    """
    labels_t = torch.as_tensor(np.array(labels, dtype=np.int64))
    weights = (1.0 / torch.bincount(labels_t).float())[labels_t]
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(weights.double(), num_samples=len(labels_t), replacement=True, generator=gen)
