"""Backbone registry.

All backbones map (B, C_in, N) -> (B, N, d_f) per-point features.
Add a new encoder by decorating a factory with ``@register("Name")``.
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from .dgcnn import DGCNN
from .pointnet import PointNet, PointNetLite, TNet
from .pointnet2 import PointNetPlusPlus

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


@register("PointNet")
def _pointnet(d_f: int, in_channel: int, **_) -> nn.Module:
    return PointNet(out_channel=d_f, in_channel=in_channel)


@register("PointNetLite")
def _pointnet_lite(d_f: int, in_channel: int, width: int = 64, **_) -> nn.Module:
    return PointNetLite(out_channel=d_f, in_channel=in_channel, width=width)


@register("PointNetPlusPlus")
def _pointnet2(d_f: int, in_channel: int, **_) -> nn.Module:
    return PointNetPlusPlus(out_channel=d_f, in_channel=in_channel)


@register("DGCNN")
def _dgcnn(d_f: int, in_channel: int, k: int = 20, **_) -> nn.Module:
    return DGCNN(out_channel=d_f, in_channel=in_channel, k=k)


# Name used by the original course-project configs / checkpoints.
_REGISTRY["OfficialPointNet"] = _REGISTRY["PointNet"]


def available_backbones() -> list[str]:
    return sorted(_REGISTRY)


def build_backbone(name: str, d_f: int = 128, in_channel: int = 6, **kwargs) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {available_backbones()}")
    return _REGISTRY[name](d_f=d_f, in_channel=in_channel, **kwargs)


__all__ = [
    "DGCNN",
    "PointNet",
    "PointNetLite",
    "PointNetPlusPlus",
    "TNet",
    "available_backbones",
    "build_backbone",
    "register",
]
