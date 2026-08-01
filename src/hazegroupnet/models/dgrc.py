"""Dual-Cue Group Residual Calibration (DGRC)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _spatial_minmax(value: Tensor, eps: float = 1e-6) -> Tensor:
    low = value.amin(dim=(-2, -1), keepdim=True)
    high = value.amax(dim=(-2, -1), keepdim=True)
    return (value - low) / (high - low + eps)


def dark_channel_haze_proxy(image: Tensor, patch_size: int = 15) -> Tensor:
    """Return the bounded DCP-inspired input cue used by DGRC.

    The proxy mixes the absolute dark-channel level with its relative spatial
    distribution. It is a conditioning signal rather than a physical
    transmission estimate.
    """

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")

    channel_minimum = image.float().amin(dim=1, keepdim=True)
    dark_channel = -F.max_pool2d(
        -channel_minimum,
        kernel_size=patch_size,
        stride=1,
        padding=patch_size // 2,
    )
    absolute = dark_channel.clamp(0.0, 1.0)
    relative = _spatial_minmax(dark_channel)
    return 0.5 * absolute + 0.5 * relative


def _channel_rms_calibration(feature: Tensor, eps: float) -> Tensor:
    inverse_rms = torch.rsqrt(feature.float().square().mean(dim=1, keepdim=True) + eps)
    return feature * inverse_rms.to(feature.dtype)


def _cosine_discrepancy(
    reference: Tensor,
    candidate: Tensor,
    eps: float,
) -> Tensor:
    similarity = F.cosine_similarity(
        reference.float(),
        candidate.float(),
        dim=1,
        eps=eps,
    ).unsqueeze(1)
    return (1.0 - similarity).clamp_(0.0, 2.0)


class DualCueGroupResidualCalibration(nn.Module):
    """Calibrate native groups around a preserved additive skip pathway.

    The input-derived haze proxy and three decoder-conditioned discrepancies
    form the two cue families. Three-way routing probabilities are centered
    relative to the uniform prior. Consequently, uniform routing produces an
    exactly zero DGRC residual and recovers ``decoder + skip``.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_channels: int = 8,
        route_temperature: float = 1.25,
        layer_scale: float = 1e-2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels <= 0 or channels % 4:
            raise ValueError("channels must be a positive multiple of four")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if route_temperature <= 0:
            raise ValueError("route_temperature must be positive")
        if layer_scale < 0:
            raise ValueError("layer_scale must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.channels = int(channels)
        self.split_sizes = (channels // 4, channels // 2, channels // 4)
        self.route_temperature = float(route_temperature)
        self.eps = float(eps)

        self.decoder_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        nn.init.dirac_(self.decoder_projection.weight)

        self.router = nn.Sequential(
            nn.Conv2d(
                4,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                padding_mode="replicate",
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        nn.init.normal_(self.router[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.router[-1].bias)

        # Bias-free projection is necessary for the exact neutral-state result.
        self.output_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(layer_scale)))

    def _validate_inputs(
        self,
        decoder_feature: Tensor,
        skip_feature: Tensor,
        haze_proxy: Tensor,
        native_groups: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        if decoder_feature.ndim != 4 or decoder_feature.shape[1] != self.channels:
            raise ValueError(f"decoder_feature must have shape [B, {self.channels}, H, W]")
        if skip_feature.shape != decoder_feature.shape:
            raise ValueError("skip_feature must have the same shape as decoder_feature")
        if haze_proxy.ndim != 4 or haze_proxy.shape[:2] != (
            decoder_feature.shape[0],
            1,
        ):
            raise ValueError("haze_proxy must have shape [B, 1, H, W]")
        if not isinstance(native_groups, (tuple, list)) or len(native_groups) != 3:
            raise ValueError("native_groups must contain exactly three tensors")

        batch = decoder_feature.shape[0]
        spatial = decoder_feature.shape[-2:]
        for index, (group, channels) in enumerate(zip(native_groups, self.split_sizes)):
            expected = (batch, channels, *spatial)
            if tuple(group.shape) != expected:
                raise ValueError(
                    f"native_groups[{index}] must have shape {expected}; got {tuple(group.shape)}"
                )
            if group.device != decoder_feature.device:
                raise ValueError("all feature tensors must share a device")

    def forward(
        self,
        decoder_feature: Tensor,
        skip_feature: Tensor,
        haze_proxy: Tensor,
        *,
        native_groups: tuple[Tensor, Tensor, Tensor] | None = None,
        encoder_groups: tuple[Tensor, Tensor, Tensor] | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Fuse one decoder scale.

        ``encoder_groups`` is accepted as a checkpoint-era compatibility alias
        for ``native_groups``.
        """

        if native_groups is None:
            native_groups = encoder_groups
        elif encoder_groups is not None:
            raise ValueError("pass only one of native_groups or encoder_groups")
        if native_groups is None:
            raise ValueError("native_groups are required")

        self._validate_inputs(
            decoder_feature,
            skip_feature,
            haze_proxy,
            native_groups,
        )
        height, width = decoder_feature.shape[-2:]

        with torch.amp.autocast(decoder_feature.device.type, enabled=False):
            aligned_decoder = self.decoder_projection(decoder_feature.detach().float())
        decoder_groups = torch.split(
            aligned_decoder,
            self.split_sizes,
            dim=1,
        )
        discrepancies = tuple(
            _cosine_discrepancy(group.detach(), decoder_group, self.eps)
            for group, decoder_group in zip(native_groups, decoder_groups)
        )
        haze = F.interpolate(
            haze_proxy.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        route_cues = torch.cat((haze, *discrepancies), dim=1)

        with torch.amp.autocast(decoder_feature.device.type, enabled=False):
            raw_logits = self.router(route_cues.float())
            bounded_logits = 4.0 * torch.tanh(raw_logits / 4.0)
            route_probabilities = torch.softmax(
                bounded_logits / self.route_temperature,
                dim=1,
            )
            route_corrections = 3.0 * (
                route_probabilities - route_probabilities.mean(dim=1, keepdim=True)
            )

            calibrated_groups = tuple(
                _channel_rms_calibration(group.float(), self.eps) for group in native_groups
            )
            corrected_groups = tuple(
                group * correction
                for group, correction in zip(
                    calibrated_groups,
                    route_corrections.split(1, dim=1),
                )
            )
            residual = self.output_projection(torch.cat(corrected_groups, dim=1))
            additive = decoder_feature.float() + skip_feature.float()
            output = additive + self.residual_scale.float() * residual

        if not return_aux:
            return output

        branch_rms = torch.cat(
            tuple(
                group.float().square().mean(dim=1, keepdim=True).sqrt()
                for group in calibrated_groups
            ),
            dim=1,
        )
        input_rms = torch.cat(
            tuple(
                group.float().square().mean(dim=1, keepdim=True).sqrt() for group in native_groups
            ),
            dim=1,
        )
        return output, {
            "route_probabilities": route_probabilities.detach(),
            "route_corrections": route_corrections.detach(),
            "route_multipliers": (1.0 + route_corrections).detach(),
            "route_logits": bounded_logits.detach(),
            "haze_proxy": haze.detach(),
            "group_discrepancies": torch.cat(discrepancies, dim=1).detach(),
            "branch_rms": branch_rms.detach(),
            "input_rms": input_rms.detach(),
            "calibration_residual": residual.detach(),
        }


# Backward-compatible class name used by the private experiment workspace.
GroupAwareDualCueRouting = DualCueGroupResidualCalibration


__all__ = [
    "DualCueGroupResidualCalibration",
    "GroupAwareDualCueRouting",
    "dark_channel_haze_proxy",
]
