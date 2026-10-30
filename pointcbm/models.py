"""PC-CBM: a concept bottleneck model for 3D point clouds.

    points (B, N, C) ──► backbone f ──► per-point features F  (B, N, d_f)
                                              │
    concept text embeddings E (K, d_t) ──► CCAM g: sigmoid(<W_f F, W_t E> / sqrt(d))
                                              │
                                   point-concept attention A  (B, N, K)
                                              │  top-k log-sum-exp pooling over points
                                   concept scores C_hat       (B, K)
                                              │
                               linear predictor h (K -> classes)  ──► logits

Every prediction is a *linear* function of K human-readable concept scores, and
each score can be traced back to the points that produced it via ``A``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CCAM(nn.Module):
    """Concept Cross-Attention Module: per-point, per-concept sigmoid attention."""

    def __init__(self, d_f: int, d_t: int, d_hidden: int):
        super().__init__()
        self.norm_f = nn.LayerNorm(d_f)
        self.norm_t = nn.LayerNorm(d_t)
        self.proj_features = nn.Linear(d_f, d_hidden)
        self.proj_text = nn.Linear(d_t, d_hidden)
        self.scale = d_hidden**-0.5

    def forward(self, feats: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """feats (B, N, d_f), text (K, d_t) -> attention A in [0, 1], shape (B, N, K)."""
        q = self.proj_features(self.norm_f(feats))  # (B, N, h)
        k = self.proj_text(self.norm_t(text))  # (K, h)
        return torch.sigmoid(q @ k.T * self.scale)


def pool_concepts(A: torch.Tensor, method: str, beta: torch.Tensor, topk_frac: float = 0.1) -> torch.Tensor:
    """Aggregate point-level attention (B, N, K) into concept scores (B, K)."""
    if method == "mean":
        return A.mean(dim=1)
    if method == "max":
        return A.max(dim=1)[0]
    if method in ("lse", "topk_lse"):
        if method == "topk_lse":
            k = max(1, int(A.shape[1] * topk_frac))
            A = torch.topk(A, k=k, dim=1)[0]
        b = torch.clamp(beta, min=1.0, max=50.0)
        return torch.logsumexp(b * A, dim=1) / b
    raise ValueError(f"Unknown pooling '{method}'")


class ConceptBottleneck3D(nn.Module):
    """Backbone + CCAM + transparent linear predictor.

    Args:
        backbone: module mapping (B, C_in, N) -> (B, N, d_f).
        pooling: ``topk_lse`` (default, smooth max over the top 10% of points),
            ``lse``, ``max`` or ``mean``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_f: int,
        d_t: int,
        d_hidden: int,
        num_concepts: int,
        num_classes: int,
        task: str = "classification",
        beta: float = 10.0,
        pooling: str = "topk_lse",
        topk_frac: float = 0.1,
        concept_dropout: float = 0.3,
    ):
        super().__init__()
        self.task = task
        self.pooling = pooling
        self.topk_frac = topk_frac
        self.backbone = backbone
        self.feat_norm = nn.LayerNorm(d_f)
        self.ccam = CCAM(d_f, d_t, d_hidden)
        self.dropout = nn.Dropout(p=concept_dropout)
        self.predictor = nn.Linear(num_concepts, num_classes)
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def concept_attention(self, X: torch.Tensor, E_text: torch.Tensor) -> torch.Tensor:
        feats = self.feat_norm(self.backbone(X.transpose(1, 2)))
        return self.ccam(feats, E_text)

    def predict_from_concepts(self, C: torch.Tensor) -> torch.Tensor:
        """Run only the transparent head — used for test-time concept interventions."""
        return self.predictor(C)

    def forward(self, X: torch.Tensor, E_text: torch.Tensor):
        """
        Args:
            X: point clouds (B, N, C_in)
            E_text: frozen concept text embeddings (K, d_t)
        Returns:
            logits (B, num_classes), concept scores C_hat (B, K), attention A (B, N, K)
        """
        A = self.concept_attention(X, E_text)
        C_hat = pool_concepts(A, self.pooling, self.beta, self.topk_frac)
        if self.task == "classification":
            logits = self.predictor(self.dropout(C_hat))
        elif self.task == "segmentation":
            logits = self.predictor(A)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        return logits, C_hat, A


class BlackBoxClassifier(nn.Module):
    """Same backbone, no bottleneck: [max ‖ mean] pooled features -> MLP.

    The accuracy gap between this and :class:`ConceptBottleneck3D` is the
    "price of interpretability" reported in the ablations.
    """

    def __init__(self, backbone: nn.Module, d_f: int, d_hidden: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(2 * d_f, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, X: torch.Tensor, E_text: torch.Tensor | None = None):
        feats = self.backbone(X.transpose(1, 2))  # (B, N, d_f)
        pooled = torch.cat([feats.max(dim=1)[0], feats.mean(dim=1)], dim=-1)
        return self.head(pooled), None, None


def random_concept_bank(num_concepts: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Unit-norm Gaussian vectors with the same shape as a CLIP concept bank.

    Control condition: if random "concepts" match CLIP concepts, the language
    grounding is not what drives accuracy.
    """
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(num_concepts, dim, generator=g)
    return E / E.norm(dim=-1, keepdim=True)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def build_model(cfg: dict, d_t: int, num_concepts: int) -> nn.Module:
    """Construct a model from the ``model`` section of a config dict."""
    from .backbones import build_backbone

    m = cfg["model"]
    backbone = build_backbone(
        m["backbone"], d_f=m["d_f"], in_channel=m.get("in_channel", 6), k=m.get("k", 20), width=m.get("width", 64)
    )
    arch = m.get("head", "cbm")
    if arch == "cbm":
        return ConceptBottleneck3D(
            backbone=backbone,
            d_f=m["d_f"],
            d_t=d_t,
            d_hidden=m["d_hidden"],
            num_concepts=num_concepts,
            num_classes=m.get("num_classes", 2),
            task=cfg.get("experiment", {}).get("task", "classification"),
            pooling=m.get("pooling", "topk_lse"),
            topk_frac=m.get("topk_frac", 0.1),
            beta=m.get("beta", 10.0),
        )
    if arch == "blackbox":
        return BlackBoxClassifier(backbone, d_f=m["d_f"], d_hidden=m["d_hidden"], num_classes=m.get("num_classes", 2))
    raise ValueError(f"Unknown model.head '{arch}' (expected 'cbm' or 'blackbox')")


__all__ = [
    "CCAM",
    "BlackBoxClassifier",
    "ConceptBottleneck3D",
    "build_model",
    "count_parameters",
    "pool_concepts",
    "random_concept_bank",
]
