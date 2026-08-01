"""Per-image RGB full-reference metrics in the [0, 1] range."""

from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] not in {1, 3}:
        raise ValueError(f"Expected HxWx1 or HxWx3 image, received {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    else:
        image = image.astype(np.float64)
    return np.clip(image, 0.0, 1.0)


def _validate_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = _as_float_rgb(prediction), _as_float_rgb(target)
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} != target shape {gt.shape}")
    return pred, gt


def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return RGB PSNR calculated per image with a data range of one."""
    pred, gt = _validate_pair(prediction, target)
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return the frozen Gaussian 11x11 RGB SSIM used in the paper.

    The protocol uses sigma 1.5, population covariance, and the valid
    interior after filtering.
    """

    pred, gt = _validate_pair(prediction, target)
    min_size = min(pred.shape[:2])
    if min_size < 11:
        raise ValueError("SSIM requires images at least 11x11 pixels")
    return float(
        structural_similarity(
            gt,
            pred,
            data_range=1.0,
            channel_axis=-1,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
        )
    )


def delta_e00(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return mean CIEDE2000 color difference over RGB pixels."""
    pred, gt = _validate_pair(prediction, target)
    if pred.shape[-1] != 3:
        raise ValueError("CIEDE2000 is defined here only for RGB images")
    return float(np.mean(deltaE_ciede2000(rgb2lab(pred), rgb2lab(gt))))


def compute_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute the repository's standard per-image restoration metrics."""
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "delta_e00": delta_e00(prediction, target),
    }
