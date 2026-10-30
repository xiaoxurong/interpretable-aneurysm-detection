"""Training objectives for PC-CBM.

total = focal(logits, y)
      + λ_sparse · H(A)            attention entropy  -> localised concept maps
      + λ_ortho  · cos(A_i, A_j)   overlap penalty    -> concepts look at different regions
      + λ_supcon · SupCon(C_hat)   class-structured concept space
      + λ_align  · BCE(C_hat, C*)  optional alignment to pseudo-labels from a 3D-VLM
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Class-weighted focal loss (Lin et al., 2017) on raw logits.

    alpha weights the positive (aneurysm) class; gamma down-weights easy examples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha).to(logits.dtype)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


class SparsityLoss(nn.Module):
    """Mean Shannon entropy of each concept's attention distribution over points."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        p = A / (A.sum(dim=1, keepdim=True) + self.eps)
        return -(p * torch.log(p + self.eps)).sum(dim=1).mean()


class OrthogonalityLoss(nn.Module):
    """Mean |cosine similarity| between the spatial attention maps of different concepts."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        B, _, K = A.shape
        if K < 2:
            return A.new_zeros(())
        A_n = A / (A.norm(dim=1, keepdim=True) + self.eps)
        sim = torch.bmm(A_n.transpose(1, 2), A_n)  # (B, K, K)
        off_diag = 1 - torch.eye(K, device=A.device, dtype=A.dtype)
        return (sim.abs() * off_diag).sum() / (B * K * (K - 1))


class SemanticAlignmentLoss(nn.Module):
    """BCE to soft concept pseudo-labels, ignoring uncertain targets in (0.2, 0.8)."""

    def forward(self, C_hat: torch.Tensor, C_target: torch.Tensor) -> torch.Tensor:
        mask = ((C_target > 0.8) | (C_target < 0.2)).float()
        bce = F.binary_cross_entropy(C_hat.clamp(1e-6, 1 - 1e-6), C_target, reduction="none")
        return (bce * mask).mean()


class SupervisedContrastiveLoss(nn.Module):
    """SupCon (Khosla et al., NeurIPS 2020) on L2-normalised concept vectors."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        z = F.normalize(z, dim=1)
        sim = z @ z.T / self.temperature
        labels = labels.view(-1, 1)
        eye = torch.eye(B, device=z.device, dtype=z.dtype)
        positives = (labels == labels.T).to(z.dtype) - eye
        not_self = 1 - eye

        sim = sim - (sim - 1e9 * eye).max(dim=1, keepdim=True)[0].detach()
        log_denom = torch.log((torch.exp(sim) * not_self).sum(dim=1, keepdim=True) + 1e-8)
        log_prob = sim - log_denom

        n_pos = positives.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return z.new_zeros(())
        mean_log_prob_pos = (log_prob * positives).sum(dim=1) / n_pos.clamp(min=1)
        return -mean_log_prob_pos[valid].mean()


@dataclass
class LossOutput:
    total: torch.Tensor
    task: torch.Tensor
    align: torch.Tensor
    sparse: torch.Tensor
    ortho: torch.Tensor
    supcon: torch.Tensor

    def as_floats(self) -> dict[str, float]:
        return {k: float(v.detach()) for k, v in self.__dict__.items()}


class CBMLoss(nn.Module):
    """Weighted sum of the objectives above. Terms with λ = 0 are skipped entirely."""

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        lambda_align: float = 0.0,
        lambda_sparse: float = 0.01,
        lambda_ortho: float = 0.01,
        lambda_supcon: float = 0.01,
        temperature: float = 0.1,
        lambda_l1: float = 1e-4,
    ):
        super().__init__()
        self.lambdas = dict(align=lambda_align, sparse=lambda_sparse, ortho=lambda_ortho, supcon=lambda_supcon)
        self.lambda_l1 = lambda_l1
        self.task_criterion = FocalLoss(alpha=alpha, gamma=gamma)
        self.align_criterion = SemanticAlignmentLoss()
        self.sparse_criterion = SparsityLoss()
        self.ortho_criterion = OrthogonalityLoss()
        self.supcon_criterion = SupervisedContrastiveLoss(temperature=temperature)

    @classmethod
    def from_config(cls, lw: dict) -> CBMLoss:
        return cls(
            alpha=lw.get("alpha", 0.75),
            gamma=lw.get("gamma", 2.0),
            lambda_align=lw.get("lambda_align", 0.0),
            lambda_sparse=lw.get("lambda_sparse", 0.01),
            lambda_ortho=lw.get("lambda_ortho", 0.01),
            lambda_supcon=lw.get("lambda_supcon", 0.01),
            temperature=lw.get("temperature", 0.1),
            lambda_l1=lw.get("lambda_l1", 1e-4),
        )

    def forward(self, logits, targets, C_hat=None, A=None, C_target=None, predictor_weight=None) -> LossOutput:
        zero = logits.new_zeros(())
        task = self.task_criterion(logits, targets)
        align = sparse = ortho = supcon = zero
        if C_hat is not None:
            if self.lambdas["align"] > 0 and C_target is not None:
                align = self.align_criterion(C_hat, C_target)
            if self.lambdas["supcon"] > 0:
                supcon = self.supcon_criterion(C_hat, targets)
        if A is not None:
            if self.lambdas["sparse"] > 0:
                sparse = self.sparse_criterion(A)
            if self.lambdas["ortho"] > 0:
                ortho = self.ortho_criterion(A)
        total = (
            task
            + self.lambdas["align"] * align
            + self.lambdas["sparse"] * sparse
            + self.lambdas["ortho"] * ortho
            + self.lambdas["supcon"] * supcon
        )
        if predictor_weight is not None and self.lambda_l1 > 0:
            # L1 on the K->classes head keeps the explanation sparse (few concepts per class).
            total = total + self.lambda_l1 * predictor_weight.abs().sum()
        return LossOutput(total, task, align, sparse, ortho, supcon)


# Backwards-compatible name used by the original training script.
CCAMLoss = CBMLoss
