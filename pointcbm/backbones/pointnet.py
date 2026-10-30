"""PointNet-family per-point encoders.

Every backbone in :mod:`pointcbm.backbones` follows the same contract so that
the concept bottleneck can attend over individual points:

    input  : (B, C_in, N)   C_in = 3 (xyz) or 6 (xyz + normals)
    output : (B, N, d_f)    per-point features — **no global pooling**
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial transformer that predicts a k x k alignment matrix (Qi et al., 2017)."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Residual parameterisation around the identity keeps early training stable.
        iden = torch.eye(self.k, device=x.device, dtype=x.dtype).expand(B, -1, -1)
        return x.view(B, self.k, self.k) + iden


class PointNet(nn.Module):
    """PointNet with input + feature T-Nets, returning per-point features.

    The original global max-pool is removed so the concept bottleneck can attend
    over all points; a 1x1 conv projects the 1024-d per-point descriptors to
    ``out_channel``.

    Parameter names are kept identical to the original ``OfficialPointNet`` so
    checkpoints trained with the course-project code still load.
    """

    def __init__(self, out_channel: int = 128, in_channel: int = 6):
        super().__init__()
        self.stn = TNet(k=in_channel)  # input transform
        self.fstn = TNet(k=64)  # feature transform

        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.proj = nn.Sequential(
            nn.Conv1d(1024, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        # Exposed so the feature-transform regulariser can read it after forward().
        self.last_feature_transform: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trans = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.fstn(x)
        self.last_feature_transform = trans_feat
        x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        x = self.proj(x)  # (B, out_channel, N)
        return x.transpose(1, 2).contiguous()


class PointNetLite(nn.Module):
    """Compact shared-MLP encoder (no T-Nets) — ~25x fewer FLOPs than :class:`PointNet`.

    Used for the CPU-budget ablations and as a latency-friendly deployment option.
    A global context vector (max-pooled) is concatenated back to every point so the
    per-point features are not purely local.
    """

    def __init__(self, out_channel: int = 128, in_channel: int = 6, width: int = 64):
        super().__init__()
        w = width
        self.local = nn.Sequential(
            nn.Conv1d(in_channel, w, 1, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.Conv1d(w, w, 1, bias=False),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.Conv1d(w, 2 * w, 1, bias=False),
            nn.BatchNorm1d(2 * w),
            nn.ReLU(inplace=True),
        )
        self.globl = nn.Sequential(
            nn.Conv1d(2 * w, 4 * w, 1, bias=False),
            nn.BatchNorm1d(4 * w),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(2 * w + 4 * w, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)  # (B, 2w, N)
        g = self.globl(local).max(dim=2, keepdim=True)[0]  # (B, 4w, 1)
        g = g.expand(-1, -1, local.shape[2])
        out = self.fuse(torch.cat([local, g], dim=1))
        return out.transpose(1, 2).contiguous()
