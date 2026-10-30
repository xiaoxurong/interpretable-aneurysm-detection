"""PointNet++ (SSG) encoder-decoder producing per-point features.

Encoder: three set-abstraction (SA) levels 2048 -> 512 -> 128 -> 32 centroids.
Decoder: three feature-propagation (FP) levels interpolate back to all N points,
U-Net style, so every input point gets a multi-scale descriptor.

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point
Sets in a Metric Space", NeurIPS 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance. src (B, N, 3), dst (B, M, 3) -> (B, N, M)."""
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src**2, -1).unsqueeze(-1)
    dist += torch.sum(dst**2, -1).unsqueeze(1)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather ``points`` (B, N, C) at ``idx`` (B, S) or (B, S, K)."""
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int, random_start: bool = True) -> torch.Tensor:
    """Iterative farthest-point sampling. xyz (B, N, 3) -> indices (B, npoint).

    ``random_start=False`` (used at eval time) makes inference deterministic.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    if random_start:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    else:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Indices of up to ``nsample`` points within ``radius`` of each centroid (padded)."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, :1].expand(-1, -1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        # xyz: (B, 3, N), points: (B, D, N) or None
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = (xyz - new_xyz).unsqueeze(1)
            grouped_points = grouped_xyz if points is None else torch.cat([grouped_xyz, points.unsqueeze(1)], -1)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint, random_start=self.training)
            new_xyz = index_points(xyz, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
            if points is not None:
                grouped_points = torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
            else:
                grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, D, nsample, npoint)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]  # max over the local neighbourhood
        return new_xyz.permute(0, 2, 1), new_points


class PointNetFeaturePropagation(nn.Module):
    """Inverse-distance-weighted 3-NN interpolation + skip connection + shared MLP."""

    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """xyz1 (B,3,N) fine, xyz2 (B,3,S) coarse, points1 (B,D,N)|None, points2 (B,D',S)."""
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]

        if S == 1:
            interpolated = points2.expand(-1, N, -1)
        else:
            dists, idx = square_distance(xyz1, xyz2).topk(3, dim=-1, largest=False)
            weight = 1.0 / (dists + 1e-8)
            weight = weight / weight.sum(dim=2, keepdim=True)
            interpolated = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1.permute(0, 2, 1), interpolated], dim=-1)
        else:
            new_points = interpolated

        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetPlusPlus(nn.Module):
    """Segmentation-style PointNet++: (B, C_in, N) -> (B, N, out_channel)."""

    def __init__(self, out_channel: int = 128, in_channel: int = 6):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # sa1 sees relative xyz (3) + extra per-point features (in_channel - 3)
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128])
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = PointNetSetAbstraction(32, 0.8, 64, 256 + 3, [256, 256, 512])

        self.fp3 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 128, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, out_channel])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l0_xyz = x[:, :3, :]
        l0_points = x[:, 3:, :] if self.in_channel > 3 else None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        return l0_points.transpose(1, 2).contiguous()
