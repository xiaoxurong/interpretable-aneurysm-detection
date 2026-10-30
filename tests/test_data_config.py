import os

import numpy as np
import pandas as pd
import pytest
from conftest import DATA, requires_data

from pointcbm.config import load_config
from pointcbm.data import IntrAClassificationDataset, fold_csv, random_rotation_matrix, subject_id


@requires_data
@pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
def test_splits_are_subject_disjoint(fold):
    """No subject may contribute segments to more than one of train / val / test."""
    subj = {s: set(pd.read_csv(fold_csv(DATA, fold, s)).filename.map(subject_id)) for s in ("train", "val", "test")}
    assert not subj["train"] & subj["test"]
    assert not subj["train"] & subj["val"]
    assert not subj["val"] & subj["test"]


@requires_data
def test_test_folds_partition_the_dataset():
    tests = [set(pd.read_csv(fold_csv(DATA, k, "test")).filename) for k in range(1, 6)]
    assert sum(map(len, tests)) == len(set.union(*tests)) == len(pd.read_csv(os.path.join(DATA, "all_labels.csv")))


@requires_data
def test_dataset_item_and_subsampling():
    ds = IntrAClassificationDataset(fold_csv(DATA, 1, "val"), os.path.join(DATA, "points"), num_points=256)
    X, y = ds[0]
    assert X.shape == (256, 6) and y.item() in (0, 1)
    X2, _ = ds[0]
    assert (X == X2).all(), "eval-time subsampling must be deterministic"


@requires_data
def test_rotation_augmentation_keeps_unit_normals():
    ds = IntrAClassificationDataset(fold_csv(DATA, 1, "val"), os.path.join(DATA, "points"), augment=True, rotate=True)
    X, _ = ds[0]
    norms = X[:, 3:].norm(dim=1)
    assert np.allclose(norms.numpy(), 1.0, atol=1e-3)


def test_random_rotation_is_proper():
    R = random_rotation_matrix(np.random.default_rng(0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-8) and np.isclose(np.linalg.det(R), 1.0)


def test_subject_id_parsing():
    assert subject_id("aneurysm_ArteryObjAN117-2.npy") == "AN117"
    with pytest.raises(ValueError):
        subject_id("no_subject_here.npy")


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  backbone: DGCNN\n")
    cfg = load_config(str(p), ["training.num_epochs=3", "model.head=blackbox", "data.num_points=null"])
    assert cfg["model"]["backbone"] == "DGCNN" and cfg["model"]["head"] == "blackbox"
    assert cfg["training"]["num_epochs"] == 3 and cfg["data"]["num_points"] is None
    assert cfg["training"]["batch_size"] == 128  # default retained
