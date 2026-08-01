"""Configuration loading and model construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from hazegroupnet.models import HAZEGROUP_VARIANTS, HazeGroupNet, create_model


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and validate its top-level structure."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    for section in ("experiment", "model", "data", "train", "evaluation"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"missing mapping section {section!r}")
    return payload


def _tuple_of_ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of integers")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integers") from error
    return result


def validate_model_recipe(model_config: Mapping[str, Any]) -> str:
    """Validate redundant architecture fields against the named variant."""

    variant = str(model_config.get("variant", "small")).strip().lower()
    aliases = {
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
    key = aliases.get(variant)
    if key is None:
        raise ValueError(f"unknown model variant: {variant!r}")
    recipe = HAZEGROUP_VARIANTS[key]

    checks: tuple[tuple[str, object], ...] = (
        ("base_channels", recipe.base_channels),
        ("encoder_depths", recipe.encoder_depths),
        ("bottleneck_depth", recipe.bottleneck_depth),
        ("decoder_depths", recipe.decoder_depths),
    )
    for field, expected in checks:
        if field not in model_config:
            continue
        observed: object = model_config[field]
        if isinstance(expected, tuple):
            observed = _tuple_of_ints(observed, field)
        else:
            observed = int(observed)
        if observed != expected:
            raise ValueError(f"{field}={observed!r} conflicts with the {key} recipe ({expected!r})")
    return key


def create_model_from_config(config: Mapping[str, Any]) -> HazeGroupNet:
    """Construct HazeGroupNet from a complete configuration mapping."""

    raw_model = config.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("config['model'] must be a mapping")
    variant = validate_model_recipe(raw_model)
    kwargs = {
        "density_patch_size": int(raw_model.get("density_patch_size", 15)),
        "gate_hidden_channels": int(raw_model.get("gate_hidden_channels", 8)),
        "route_temperature": float(raw_model.get("route_temperature", 1.25)),
    }
    if "dgrc_layer_scale" in raw_model:
        kwargs["dgrc_layer_scale"] = float(raw_model["dgrc_layer_scale"])
    return create_model(variant, **kwargs)


__all__ = [
    "create_model_from_config",
    "load_config",
    "validate_model_recipe",
]
