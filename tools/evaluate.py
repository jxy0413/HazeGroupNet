"""Evaluate a HazeGroupNet checkpoint on a paired CSV manifest.

Example:
    python tools/evaluate.py --config configs/rrshid/small.yaml \
        --checkpoint path/to/weights.pt --dataset-root /path/to/dataset \
        --manifest splits/test.csv --output-dir results/test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hazegroupnet.data import read_manifest  # noqa: E402
from hazegroupnet.metrics import compute_metrics  # noqa: E402
from hazegroupnet.utils import load_model_checkpoint  # noqa: E402
from hazegroupnet.utils.config import (  # noqa: E402
    create_model_from_config,
    load_config,
)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB").save(
        path
    )


def mean_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = {key: [float(row[key]) for row in rows] for key in ("psnr", "ssim", "delta_e00")}
    return {"num_images": len(rows), **{key: float(np.mean(item)) for key, item in values.items()}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="Path to a PyTorch checkpoint."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root used to resolve manifest-relative paths.",
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="CSV containing image_id,hazy_path,gt_path."
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        required=True,
        type=Path,
        help="Directory for per_image.csv and summary.json.",
    )
    parser.add_argument(
        "--split", default="", help="Optionally evaluate only rows with this split label."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Write 8-bit PNG predictions under output/predictions.",
    )
    parser.add_argument(
        "--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    records = read_manifest(args.manifest, args.dataset_root)
    if args.split:
        records = [record for record in records if record.split == args.split]
        if not records:
            raise SystemExit(f"No manifest rows matched --split {args.split!r}")
    model = create_model_from_config(config).to(device).eval()
    missing, unexpected = load_model_checkpoint(
        model, args.checkpoint, map_location=device, strict=not args.non_strict
    )
    if args.non_strict and (missing or unexpected):
        print(
            json.dumps({"missing_keys": missing, "unexpected_keys": unexpected}, indent=2),
            file=sys.stderr,
        )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for index, record in enumerate(records, start=1):
            hazy = load_rgb(record.hazy_path)
            target = load_rgb(record.gt_path)
            if hazy.shape != target.shape:
                raise ValueError(
                    f"{record.image_id}: hazy {hazy.shape} and reference {target.shape} differ"
                )
            prediction = to_image(model(to_tensor(hazy, device)))
            metrics = compute_metrics(prediction, target)
            row: dict[str, Any] = {
                "image_id": record.image_id,
                "split": record.split,
                "haze_level": record.haze_level,
                **metrics,
            }
            rows.append(row)
            if args.save_predictions:
                save_rgb(
                    prediction,
                    args.output_dir / "predictions" / f"{record.image_id}.png",
                )
            print(
                f"[{index}/{len(records)}] {record.image_id}: "
                f"PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.6f}, "
                f"DeltaE00={metrics['delta_e00']:.4f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "split", "haze_level", "psnr", "ssim", "delta_e00"]
        )
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["haze_level"]:
            grouped[str(row["haze_level"])].append(row)
    summary: dict[str, Any] = {
        "variant": model.recipe.name,
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "overall": mean_summary(rows),
        "by_haze_level": {
            level: mean_summary(level_rows) for level, level_rows in sorted(grouped.items())
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
