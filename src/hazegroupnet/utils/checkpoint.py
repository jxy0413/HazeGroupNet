"""Checkpoint loading compatible with common PyTorch training wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

_STATE_KEYS = ("model", "state_dict", "model_state_dict")


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("A checkpoint must be a state_dict or a dictionary containing one")
    for key in _STATE_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if (
        payload
        and all(isinstance(key, str) for key in payload)
        and all(torch.is_tensor(value) for value in payload.values())
    ):
        return payload  # type: ignore[return-value]
    raise KeyError(f"Checkpoint does not contain a state_dict under any of {_STATE_KEYS}")


def _strip_distributed_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """Load a checkpoint and return a normalized, non-DataParallel state dict."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return _strip_distributed_prefix(_extract_state_dict(payload))


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a model checkpoint and return missing/unexpected key lists.

    ``strict=True`` is the safe default for reported experiments. Set it to
    false only when intentionally adapting an architecture.
    """
    result = model.load_state_dict(load_checkpoint(path, map_location), strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
