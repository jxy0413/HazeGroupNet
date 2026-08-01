from __future__ import annotations

import pytest
import torch


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("tiny", 545_007), ("small", 1_707_975), ("large", 4_071_199)],
)
def test_model_variants_have_expected_parameter_count(
    variant: str,
    expected: int,
) -> None:
    from hazegroupnet.models import create_model

    model = create_model(variant)
    assert _count_parameters(model) == expected


def test_model_preserves_odd_input_shape() -> None:
    from hazegroupnet.models import create_model

    model = create_model("tiny").eval()
    image = torch.rand(1, 3, 67, 71)
    with torch.inference_mode():
        prediction = model(image)
    assert prediction.shape == image.shape


def test_uniform_routes_produce_zero_calibration_residual() -> None:
    from hazegroupnet.models import DualCueGroupResidualCalibration

    module = DualCueGroupResidualCalibration(channels=16).eval()
    module.router = torch.nn.Conv2d(4, 3, kernel_size=1, bias=False)
    torch.nn.init.zeros_(module.router.weight)
    decoder = torch.randn(2, 16, 9, 11)
    skip = torch.randn(2, 16, 9, 11)
    groups = tuple(torch.randn(2, width, 9, 11) for width in (4, 8, 4))
    haze_proxy = torch.rand(2, 1, 9, 11)

    with torch.inference_mode():
        output, aux = module(
            decoder,
            skip,
            haze_proxy,
            native_groups=groups,
            return_aux=True,
        )
    assert torch.allclose(
        aux["route_probabilities"], torch.full_like(aux["route_probabilities"], 1 / 3)
    )
    assert torch.allclose(
        aux["calibration_residual"], torch.zeros_like(aux["calibration_residual"]), atol=1e-7
    )
    assert torch.allclose(output, decoder + skip, atol=1e-7, rtol=0.0)


def test_frozen_early_stopping_starts_after_full_patience_window() -> None:
    from hazegroupnet.utils import EarlyStoppingConfig, EarlyStoppingState

    config = EarlyStoppingConfig(
        eligibility_step=10_000,
        earliest_stop_step=12_500,
        patience=5,
    )
    config.validate(max_steps=15_000, validation_interval=500)
    state = EarlyStoppingState()

    decisions = []
    for index, step in enumerate(range(10_000, 12_501, 500)):
        decisions.append(
            state.observe(
                step=step,
                psnr=24.0 - 0.001 * index,
                ssim=0.90,
                config=config,
            )
        )

    assert decisions[0]["established_anchor"]
    assert not any(item["should_stop"] for item in decisions[:-1])
    assert decisions[-1]["bad_eligible_evaluations"] == 5
    assert decisions[-1]["should_stop"]
