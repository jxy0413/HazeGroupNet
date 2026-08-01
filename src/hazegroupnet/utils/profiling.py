"""Minimal convolutional-MAC profiler used by the release scripts."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def count_convolutional_macs(
    model: nn.Module,
    input_tensor: Tensor,
) -> int:
    """Count Conv2d multiply-accumulate operations for one forward pass.

    Pooling, interpolation, normalization, activation, element-wise
    arithmetic, and the dark-channel proxy are deliberately excluded. This
    matches the *convolutional MACs* convention stated in the manuscript.
    """

    total_macs = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal total_macs
        if not isinstance(module, nn.Conv2d):
            return
        if not isinstance(output, Tensor):
            raise TypeError("Conv2d output must be a Tensor")
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        operations_per_output = kernel_height * kernel_width * module.in_channels // module.groups
        total_macs += batch * out_channels * out_height * out_width * operations_per_output

    register: Callable[[nn.Module], None]

    def register(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))

    model.apply(register)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(input_tensor)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return int(total_macs)


__all__ = ["count_convolutional_macs", "count_parameters"]
