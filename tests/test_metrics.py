from __future__ import annotations

import numpy as np
import pytest


def test_identity_image_metrics_are_optimal() -> None:
    try:
        from hazegroupnet.metrics import delta_e00, psnr, ssim
    except ImportError as error:
        pytest.skip(f"metric API is not available yet: {error}")

    image = np.random.default_rng(2026).uniform(0.05, 0.95, (24, 28, 3)).astype(np.float32)
    assert np.isinf(psnr(image, image))
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    assert delta_e00(image, image) == pytest.approx(0.0, abs=1e-6)
