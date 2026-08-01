"""Tri-Receptive Blocks used by the HazeGroupNet encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ChannelRMSNorm(nn.Module):
    """RMS-normalize every spatial position along the channel dimension."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, feature: Tensor) -> Tensor:
        inverse_rms = torch.rsqrt(feature.float().square().mean(dim=1, keepdim=True) + self.eps)
        return feature * inverse_rms.to(feature.dtype) * self.weight


class TriReceptiveBlock(nn.Module):
    """Model local, medium-range, and contextual receptive-field groups."""

    def __init__(self, channels: int, layer_scale: float = 1e-2) -> None:
        super().__init__()
        if channels <= 0 or channels % 4:
            raise ValueError("channels must be a positive multiple of four")
        if layer_scale < 0:
            raise ValueError("layer_scale must be non-negative")

        local_channels = channels // 4
        medium_channels = channels // 2
        context_channels = channels // 4
        self.split_sizes = (
            local_channels,
            medium_channels,
            context_channels,
        )

        self.norm1 = ChannelRMSNorm(channels)
        self.input_mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.local_branch = nn.Conv2d(
            local_channels,
            local_channels,
            kernel_size=3,
            padding=1,
            groups=local_channels,
        )
        self.medium_branch1 = nn.Conv2d(
            medium_channels,
            medium_channels,
            kernel_size=3,
            padding=1,
            groups=medium_channels,
        )
        self.medium_branch2 = nn.Conv2d(
            medium_channels,
            medium_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=medium_channels,
        )
        self.global_context = nn.Conv2d(
            context_channels,
            context_channels,
            kernel_size=5,
            padding=2,
            groups=context_channels,
        )
        self.spatial_project = nn.Conv2d(channels, channels, kernel_size=1)
        self.spatial_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale)))

        self.norm2 = ChannelRMSNorm(channels)
        self.channel_expand = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.channel_reduce = nn.Conv2d(channels, channels, kernel_size=1)
        self.channel_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(layer_scale)))

    def forward(
        self,
        feature: Tensor,
        *,
        return_groups: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        mixed = self.input_mix(self.norm1(feature))
        local, medium, context_group = torch.split(
            mixed,
            self.split_sizes,
            dim=1,
        )

        local = self.local_branch(local)
        medium = self.medium_branch2(F.gelu(self.medium_branch1(medium)))

        height, width = context_group.shape[-2:]
        context_height = max(1, height // 4)
        context_width = max(1, width // 4)
        context = F.adaptive_avg_pool2d(
            context_group,
            output_size=(context_height, context_width),
        )
        context = self.global_context(context)
        context = F.interpolate(
            context,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        context = context + context_group.mean(dim=(-2, -1), keepdim=True)
        context_group = context_group * torch.sigmoid(context)

        native_groups = (local, medium, context_group)
        spatial = self.spatial_project(torch.cat(native_groups, dim=1))
        feature = feature + self.spatial_scale * spatial

        first, second = self.channel_expand(self.norm2(feature)).chunk(2, dim=1)
        channel = self.channel_reduce(F.gelu(first) * second)
        output = feature + self.channel_scale * channel

        if return_groups:
            return output, native_groups
        return output


class TriReceptiveStage(nn.Sequential):
    """A TRB stage that can expose the final block's native groups."""

    def forward(
        self,
        feature: Tensor,
        *,
        return_last_groups: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor, Tensor]]:
        if not return_last_groups:
            return super().forward(feature)
        if len(self) == 0:
            raise RuntimeError("TriReceptiveStage must contain a block")
        for block in self[:-1]:
            feature = block(feature)
        return self[-1](feature, return_groups=True)


class SeparableDown(nn.Module):
    """Depthwise/pointwise stride-two projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = ChannelRMSNorm(in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature: Tensor) -> Tensor:
        return self.pointwise(F.gelu(self.depthwise(self.norm(feature))))


class SeparableUp(nn.Module):
    """Bilinear upsampling followed by depthwise/pointwise refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm = ChannelRMSNorm(in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, feature: Tensor, output_size: tuple[int, int]) -> Tensor:
        feature = F.interpolate(
            feature,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.pointwise(F.gelu(self.depthwise(self.norm(feature))))


def make_trb_stage(channels: int, depth: int) -> TriReceptiveStage:
    """Construct a stage while retaining numeric child names in checkpoints."""

    if depth <= 0:
        raise ValueError("stage depth must be positive")
    return TriReceptiveStage(*(TriReceptiveBlock(channels) for _ in range(depth)))


__all__ = [
    "ChannelRMSNorm",
    "SeparableDown",
    "SeparableUp",
    "TriReceptiveBlock",
    "TriReceptiveStage",
    "make_trb_stage",
]
