"""HazeGroupNet family definitions."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor, nn

from hazegroupnet.models.dgrc import (
    DualCueGroupResidualCalibration,
    dark_channel_haze_proxy,
)
from hazegroupnet.models.trb import (
    SeparableDown,
    SeparableUp,
    make_trb_stage,
)


@dataclass(frozen=True)
class HazeGroupVariant:
    """Width/depth recipe for one HazeGroupNet family member."""

    name: str
    base_channels: int
    encoder_depths: tuple[int, int, int]
    bottleneck_depth: int
    decoder_depths: tuple[int, int, int]

    @property
    def widths(self) -> tuple[int, int, int, int]:
        return tuple(self.base_channels * (2**index) for index in range(4))


HAZEGROUP_VARIANTS: dict[str, HazeGroupVariant] = {
    "tiny": HazeGroupVariant(
        name="HazeGroupNet-T",
        base_channels=16,
        encoder_depths=(2, 2, 3),
        bottleneck_depth=4,
        decoder_depths=(3, 2, 2),
    ),
    "small": HazeGroupVariant(
        name="HazeGroupNet-S",
        base_channels=24,
        encoder_depths=(3, 3, 4),
        bottleneck_depth=6,
        decoder_depths=(4, 3, 3),
    ),
    "large": HazeGroupVariant(
        name="HazeGroupNet-L",
        base_channels=32,
        encoder_depths=(4, 4, 6),
        bottleneck_depth=8,
        decoder_depths=(6, 4, 4),
    ),
}

_VARIANT_ALIASES = {
    "t": "tiny",
    "tiny": "tiny",
    "hazegroupnet-t": "tiny",
    "s": "small",
    "small": "small",
    "hazegroupnet-s": "small",
    "l": "large",
    "large": "large",
    "hazegroupnet-l": "large",
}


class HazeGroupNet(nn.Module):
    """Four-scale encoder--decoder with TRBs and DGRC skip calibration."""

    def __init__(
        self,
        recipe: HazeGroupVariant,
        *,
        density_patch_size: int = 15,
        gate_hidden_channels: int = 8,
        route_temperature: float = 1.25,
        dgrc_layer_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        if density_patch_size < 1 or density_patch_size % 2 == 0:
            raise ValueError("density_patch_size must be a positive odd integer")

        self.recipe = recipe
        self.density_patch_size = int(density_patch_size)
        c0, c1, c2, c3 = recipe.widths

        def make_dgrc(channels: int) -> DualCueGroupResidualCalibration:
            return DualCueGroupResidualCalibration(
                channels,
                hidden_channels=gate_hidden_channels,
                route_temperature=route_temperature,
                layer_scale=dgrc_layer_scale,
            )

        self.stem = nn.Conv2d(3, c0, kernel_size=3, padding=1)
        self.encoder0 = make_trb_stage(c0, recipe.encoder_depths[0])
        self.down0 = SeparableDown(c0, c1)
        self.encoder1 = make_trb_stage(c1, recipe.encoder_depths[1])
        self.down1 = SeparableDown(c1, c2)
        self.encoder2 = make_trb_stage(c2, recipe.encoder_depths[2])
        self.down2 = SeparableDown(c2, c3)
        self.bottleneck = make_trb_stage(c3, recipe.bottleneck_depth)

        self.up2 = SeparableUp(c3, c2)
        self.fusion2 = make_dgrc(c2)
        self.decoder2 = make_trb_stage(c2, recipe.decoder_depths[0])
        self.up1 = SeparableUp(c2, c1)
        self.fusion1 = make_dgrc(c1)
        self.decoder1 = make_trb_stage(c1, recipe.decoder_depths[1])
        self.up0 = SeparableUp(c1, c0)
        self.fusion0 = make_dgrc(c0)
        self.decoder0 = make_trb_stage(c0, recipe.decoder_depths[2])
        self.head = nn.Conv2d(c0, 3, kernel_size=3, padding=1)

    @staticmethod
    def _pad_to_multiple(
        image: Tensor,
        multiple: int = 8,
    ) -> tuple[Tensor, int, int]:
        height, width = image.shape[-2:]
        pad_bottom = (-height) % multiple
        pad_right = (-width) % multiple
        if pad_bottom == 0 and pad_right == 0:
            return image, 0, 0
        mode = "reflect" if pad_bottom < height and pad_right < width else "replicate"
        padded = F.pad(image, (0, pad_right, 0, pad_bottom), mode=mode)
        return padded, pad_bottom, pad_right

    def forward(
        self,
        image: Tensor,
        *,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")

        original_height, original_width = image.shape[-2:]
        padded, _, _ = self._pad_to_multiple(image)
        haze_proxy = dark_channel_haze_proxy(
            padded,
            patch_size=self.density_patch_size,
        )

        skip0, groups0 = self.encoder0(
            self.stem(padded),
            return_last_groups=True,
        )
        skip1, groups1 = self.encoder1(
            self.down0(skip0),
            return_last_groups=True,
        )
        skip2, groups2 = self.encoder2(
            self.down1(skip1),
            return_last_groups=True,
        )
        feature = self.bottleneck(self.down2(skip2))

        fusion_aux: list[dict[str, Tensor]] = []

        feature = self.up2(feature, skip2.shape[-2:])
        if return_aux:
            feature, aux2 = self.fusion2(
                feature,
                skip2,
                haze_proxy,
                native_groups=groups2,
                return_aux=True,
            )
            fusion_aux.append(aux2)
        else:
            feature = self.fusion2(
                feature,
                skip2,
                haze_proxy,
                native_groups=groups2,
            )
        feature = self.decoder2(feature)

        feature = self.up1(feature, skip1.shape[-2:])
        if return_aux:
            feature, aux1 = self.fusion1(
                feature,
                skip1,
                haze_proxy,
                native_groups=groups1,
                return_aux=True,
            )
            fusion_aux.append(aux1)
        else:
            feature = self.fusion1(
                feature,
                skip1,
                haze_proxy,
                native_groups=groups1,
            )
        feature = self.decoder1(feature)

        feature = self.up0(feature, skip0.shape[-2:])
        if return_aux:
            feature, aux0 = self.fusion0(
                feature,
                skip0,
                haze_proxy,
                native_groups=groups0,
                return_aux=True,
            )
            fusion_aux.append(aux0)
        else:
            feature = self.fusion0(
                feature,
                skip0,
                haze_proxy,
                native_groups=groups0,
            )
        feature = self.decoder0(feature)

        output = padded + self.head(feature)
        output = output[..., :original_height, :original_width]
        if not return_aux:
            return output

        # Fine-to-coarse ordering matches fusion0, fusion1, fusion2.
        return output, {
            "input_haze_proxy": haze_proxy[
                ...,
                :original_height,
                :original_width,
            ].detach(),
            "fusions": list(reversed(fusion_aux)),
        }


def build_hazegroupnet(
    variant: str = "small",
    **kwargs: object,
) -> HazeGroupNet:
    """Build a randomly initialized HazeGroupNet-T/S/L."""

    normalized = variant.strip().lower()
    key = _VARIANT_ALIASES.get(normalized)
    if key is None:
        choices = ", ".join(HAZEGROUP_VARIANTS)
        raise ValueError(f"unknown HazeGroupNet variant {variant!r}; choose {choices}")
    return HazeGroupNet(HAZEGROUP_VARIANTS[key], **kwargs)


def create_model(
    variant: str = "small",
    **kwargs: object,
) -> HazeGroupNet:
    """Public model factory."""

    return build_hazegroupnet(variant, **kwargs)


def hazegroupnet_t(**kwargs: object) -> HazeGroupNet:
    return build_hazegroupnet("tiny", **kwargs)


def hazegroupnet_s(**kwargs: object) -> HazeGroupNet:
    return build_hazegroupnet("small", **kwargs)


def hazegroupnet_l(**kwargs: object) -> HazeGroupNet:
    return build_hazegroupnet("large", **kwargs)


__all__ = [
    "HAZEGROUP_VARIANTS",
    "HazeGroupNet",
    "HazeGroupVariant",
    "build_hazegroupnet",
    "create_model",
    "hazegroupnet_l",
    "hazegroupnet_s",
    "hazegroupnet_t",
]
