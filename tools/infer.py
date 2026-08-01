"""Run a HazeGroupNet checkpoint on an RGB image or a directory of images.

Example:
    python tools/infer.py --config configs/rrshid/large.yaml \
        --checkpoint path/to/weights.pt --input path/to/image_or_directory \
        --output outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hazegroupnet.data import IMAGE_EXTENSIONS  # noqa: E402
from hazegroupnet.utils import load_model_checkpoint  # noqa: E402
from hazegroupnet.utils.config import (  # noqa: E402
    create_model_from_config,
    load_config,
)


def input_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image type: {path.suffix}")
        return [path]
    if path.is_dir():
        images = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            return images
        raise ValueError(f"No supported images found under {path}")
    raise FileNotFoundError(path)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(output, mode="RGB").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--input", required=True, type=Path, help="An RGB image or recursively scanned directory."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PNG file (single image) or output directory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    images = input_images(args.input)
    if len(images) > 1 and args.output.suffix:
        raise SystemExit("--output must be a directory when --input is a directory")

    model = create_model_from_config(config).to(device).eval()
    load_model_checkpoint(model, args.checkpoint, map_location=device, strict=not args.non_strict)
    with torch.inference_mode():
        for path in images:
            image = load_rgb(path)
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            prediction = (
                model(tensor).squeeze(0).float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            )
            if len(images) == 1 and args.output.suffix:
                destination = args.output
            else:
                destination = args.output / path.relative_to(
                    args.input if args.input.is_dir() else path.parent
                ).with_suffix(".png")
            save_rgb(prediction, destination)
            print(f"{path} -> {destination}")


if __name__ == "__main__":
    main()
