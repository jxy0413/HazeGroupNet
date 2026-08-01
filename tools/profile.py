"""Report HazeGroupNet parameters and convolutional MACs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hazegroupnet.utils.config import (  # noqa: E402
    create_model_from_config,
    load_config,
)
from hazegroupnet.utils.profiling import (  # noqa: E402
    count_convolutional_macs,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")

    config = load_config(args.config)
    model = create_model_from_config(config).cpu()
    sample = torch.zeros(1, 3, args.height, args.width)
    total, trainable = count_parameters(model)
    macs = count_convolutional_macs(model, sample)
    result = {
        "variant": model.recipe.name,
        "input_shape": [1, 3, args.height, args.width],
        "parameters": total,
        "trainable_parameters": trainable,
        "convolutional_macs": macs,
        "parameters_million": total / 1e6,
        "convolutional_macs_giga": macs / 1e9,
        "scope": (
            "Conv2d MACs only; excludes pooling, interpolation, normalization, "
            "activations, element-wise operations, and haze-proxy construction."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
