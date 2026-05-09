"""
DGCNN backbone for PC-CBM.

Output interface matches PointNetPlusPlus:
    Input : (B, in_channel, N)   — same as backbone receives inside forward()
    Output: (B, N, out_channel)  — per-point features fed into CCAM

Key difference from PointNet++:
    PointNet++ aggregates within fixed-radius ball queries (static graph).
    DGCNN recomputes the k-NN graph in *feature space* at every layer
    (hence "dynamic"), capturing both local geometry and semantic similarity.

Reference:
    Wang et al., "Dynamic Graph CNN for Learning on Point Clouds", TOG 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Core graph utilities
# ─────────────────────────────────────────────────────────────────────────────

def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors for each point in feature space.

    Args:
        x : (B, C, N)  point features
        k : number of neighbors

    Returns:
        idx : (B, N, k)  indices of k nearest neighbors
    """
    # Pairwise squared distances via the identity
    # ||a - b||^2 = ||a||^2 - 2 <a,b> + ||b||^2
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)   # (B, N, N)
    xx    = torch.sum(x ** 2, dim=1, keepdim=True)       # (B, 1, N)
    dist  = -xx - inner - xx.transpose(2, 1)             # (B, N, N) — negated so topk = nearest
    idx   = dist.topk(k=k, dim=-1)[1]                   # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20,
                      idx: torch.Tensor = None) -> torch.Tensor:
    """
    Construct edge features for EdgeConv.

    For each point x_i and each of its k neighbors x_j, the edge feature is:
        [x_i  ||  x_j - x_i]

    Args:
        x   : (B, C, N)
        k   : number of neighbors
        idx : (B, N, k) pre-computed neighbor indices (optional)

    Returns:
        edge_feat : (B, 2C, N, k)
    """
    B, C, N = x.shape
    device  = x.device

    if idx is None:
        idx = knn(x, k=k)                                  # (B, N, k)

    # Flatten indices for batch-aware gathering
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx_flat = (idx + idx_base).view(-1)                   # (B*N*k,)

    x_t       = x.transpose(2, 1).contiguous()            # (B, N, C)
    neighbors = x_t.view(B * N, C)[idx_flat]              # (B*N*k, C)
    neighbors = neighbors.view(B, N, k, C)                 # (B, N, k, C)

    center = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)  # (B, N, k, C)

    edge_feat = torch.cat([center, neighbors - center], dim=3)   # (B, N, k, 2C)
    return edge_feat.permute(0, 3, 1, 2).contiguous()            # (B, 2C, N, k)


# ─────────────────────────────────────────────────────────────────────────────
# EdgeConv block
# ─────────────────────────────────────────────────────────────────────────────

class EdgeConv(nn.Module):
    """
    Single EdgeConv layer:
        1. Build edge features  (B, 2C_in, N, k)
        2. Apply shared MLP     (B, C_out, N, k)
        3. Max-pool over k      (B, C_out, N)

    The k-NN graph is recomputed in the *current* feature space at each layer,
    which is what makes DGCNN "dynamic."
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C_in, N)
        Returns:
            (B, C_out, N)
        """
        feat = get_graph_feature(x, k=self.k)   # (B, 2C_in, N, k)
        feat = self.conv(feat)                   # (B, C_out, N, k)
        return feat.max(dim=-1)[0]              # (B, C_out, N)


# ─────────────────────────────────────────────────────────────────────────────
# DGCNN backbone
# ─────────────────────────────────────────────────────────────────────────────

class DGCNN(nn.Module):
    """
    DGCNN backbone returning per-point features for CCAM.

    Architecture:
        EdgeConv(in_channel → 64)   — local geometry in input space
        EdgeConv(64 → 64)           — local geometry in learned space
        EdgeConv(64 → 64)           — higher-level local geometry
        Cat [x1, x2, x3]            — (B, 192, N)
        Conv1d → out_channel        — per-point output features

    Args:
        out_channel : feature dimension per point (d_f), default 128.
                      Must match d_f in ConceptBottleneck3D.
        in_channel  : 3 for xyz, 6 for xyz + normals. Default 6.
        k           : number of nearest neighbors per EdgeConv. Default 20.
                      Reduce to 10 if GPU memory is tight.
    """

    def __init__(self, out_channel: int = 128,
                 in_channel: int = 6,
                 k: int = 20):
        super().__init__()
        self.k          = k
        self.in_channel = in_channel

        self.ec1 = EdgeConv(in_channel, 64, k=k)
        self.ec2 = EdgeConv(64,         64, k=k)
        self.ec3 = EdgeConv(64,         64, k=k)

        # Fuse multi-scale edge features → per-point output
        self.fusion = nn.Sequential(
            nn.Conv1d(64 * 3, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_channel, N)  — backbone receives transposed input from
                                      ConceptBottleneck3D.forward()
        Returns:
            (B, N, out_channel)     — per-point features for CCAM
        """
        x1 = self.ec1(x)                              # (B, 64, N)
        x2 = self.ec2(x1)                             # (B, 64, N)
        x3 = self.ec3(x2)                             # (B, 64, N)

        x_cat = torch.cat([x1, x2, x3], dim=1)       # (B, 192, N)
        out   = self.fusion(x_cat)                    # (B, out_channel, N)

        return out.transpose(1, 2).contiguous()        # (B, N, out_channel)