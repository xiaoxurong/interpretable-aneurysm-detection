import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from pointcbm.losses import CBMLoss, FocalLoss, OrthogonalityLoss, SparsityLoss, SupervisedContrastiveLoss
from pointcbm.metrics import binary_metrics, cluster_bootstrap_ci, expected_calibration_error, select_threshold


def test_focal_reduces_to_ce_when_gamma_zero():
    logits, y = torch.randn(16, 2), torch.randint(0, 2, (16,))
    torch.testing.assert_close(FocalLoss(alpha=0.5, gamma=0.0)(logits, y), 0.5 * F.cross_entropy(logits, y))


def test_focal_downweights_easy_examples():
    y = torch.tensor([1])
    easy, hard = torch.tensor([[-5.0, 5.0]]), torch.tensor([[1.0, -1.0]])
    f, ce = FocalLoss(gamma=2.0), lambda z: F.cross_entropy(z, y)
    assert f(easy, y) / ce(easy) < f(hard, y) / ce(hard)


def test_sparsity_is_max_for_uniform_attention():
    N = 50
    uniform = torch.ones(2, N, 3)
    peaked = torch.zeros(2, N, 3)
    peaked[:, 0] = 1
    assert math.isclose(SparsityLoss()(uniform).item(), math.log(N), rel_tol=1e-4)
    assert SparsityLoss()(peaked).item() < 1e-3


def test_orthogonality_bounds():
    disjoint = torch.zeros(1, 4, 2)
    disjoint[0, :2, 0] = 1
    disjoint[0, 2:, 1] = 1
    identical = torch.ones(1, 4, 2)
    assert OrthogonalityLoss()(disjoint).item() < 1e-6
    assert math.isclose(OrthogonalityLoss()(identical).item(), 1.0, rel_tol=1e-5)


def test_supcon_prefers_class_separated_embeddings():
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    separated = torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3) + 0.01 * torch.randn(6, 2)
    mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0]] * 3)
    loss = SupervisedContrastiveLoss(0.1)
    assert loss(separated, y) < loss(mixed, y)


def test_cbm_loss_skips_bottleneck_terms_for_blackbox():
    out = CBMLoss()(torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
    assert out.sparse == 0 and out.ortho == 0 and out.supcon == 0
    torch.testing.assert_close(out.total, out.task)


def test_threshold_selection_and_metrics():
    y = np.array([0] * 90 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.3, 90), np.linspace(0.2, 0.4, 10)])
    t = select_threshold(y, p, "youden")
    m = binary_metrics(y, p, t)
    assert m["acc1"] >= 0.8 and m["bal_acc"] > 0.8
    assert binary_metrics(y, p, 0.5)["acc1"] == 0  # the default 0.5 threshold misses every positive


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = rng.uniform(size=p.size) < p
    assert expected_calibration_error(y, p) < 0.01


def test_cluster_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 2, groups.size)
    p = np.clip(y * 0.3 + rng.uniform(size=y.size) * 0.7, 0, 1)
    est, lo, hi = cluster_bootstrap_ci(y, p, groups.astype(str), roc_auc_score, n_boot=300)
    assert lo <= est <= hi and hi - lo < 0.3
