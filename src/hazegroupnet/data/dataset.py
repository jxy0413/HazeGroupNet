"""PyTorch datasets for paired hazy/reference images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from hazegroupnet.data.manifests import ManifestRecord


def load_rgb_tensor(path: str | Path) -> Tensor:
    """Load an RGB image as a contiguous float tensor in ``[0, 1]``."""

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


class PairedImageDataset(Dataset[dict[str, Any]]):
    """Paired images with optional shared crop, flip, and 90-degree rotation."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        crop_size: int | None = None,
        augment: bool = False,
    ) -> None:
        if not records:
            raise ValueError("records must not be empty")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive")
        self.records = list(records)
        self.crop_size = crop_size
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def _shared_transform(
        self,
        hazy: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if hazy.shape != target.shape:
            raise ValueError(
                f"paired images have different shapes: "
                f"{tuple(hazy.shape)} and {tuple(target.shape)}"
            )
        if self.crop_size is not None:
            height, width = hazy.shape[-2:]
            if height < self.crop_size or width < self.crop_size:
                raise ValueError(
                    f"image size {(height, width)} is smaller than crop "
                    f"{self.crop_size}"
                )
            top = int(
                torch.randint(height - self.crop_size + 1, size=(1,)).item()
            )
            left = int(
                torch.randint(width - self.crop_size + 1, size=(1,)).item()
            )
            slices = (
                slice(None),
                slice(top, top + self.crop_size),
                slice(left, left + self.crop_size),
            )
            hazy = hazy[slices]
            target = target[slices]

        if self.augment:
            if bool(torch.randint(2, size=(1,)).item()):
                hazy = torch.flip(hazy, dims=(-1,))
                target = torch.flip(target, dims=(-1,))
            rotations = int(torch.randint(4, size=(1,)).item())
            if rotations:
                hazy = torch.rot90(hazy, rotations, dims=(-2, -1))
                target = torch.rot90(target, rotations, dims=(-2, -1))
        return hazy.contiguous(), target.contiguous()

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        hazy = load_rgb_tensor(record.hazy_path)
        target = load_rgb_tensor(record.gt_path)
        hazy, target = self._shared_transform(hazy, target)
        return {
            "hazy": hazy,
            "target": target,
            "image_id": record.image_id,
            "haze_level": record.haze_level,
        }


__all__ = ["PairedImageDataset", "load_rgb_tensor"]
