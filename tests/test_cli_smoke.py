from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _prepare_pair(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    hazy_path = dataset_root / "hazy.png"
    target_path = dataset_root / "target.png"
    dataset_root.mkdir()
    image = np.random.default_rng(2026).integers(
        0,
        256,
        size=(16, 19, 3),
        dtype=np.uint8,
    )
    Image.fromarray(image).save(hazy_path)
    Image.fromarray(image).save(target_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "hazy_path",
                "gt_path",
                "split",
                "haze_level",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "sample",
                "hazy_path": "hazy.png",
                "gt_path": "target.png",
                "split": "test",
                "haze_level": "thin",
            }
        )
    return dataset_root, manifest


def test_training_dry_run_and_evaluator(tmp_path: Path) -> None:
    from hazegroupnet.models import create_model

    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "rrshid" / "tiny.yaml"
    dataset_root, manifest = _prepare_pair(tmp_path)

    dry_run = subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "train.py"),
            "--config",
            str(config),
            "--dataset-root",
            str(dataset_root),
            "--train-manifest",
            str(manifest),
            "--val-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "run"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(dry_run.stdout)
    assert plan["effective_batch_size"] == 32
    assert plan["max_steps"] == 61_600

    checkpoint = tmp_path / "state_dict.pt"
    torch.save(create_model("tiny").state_dict(), checkpoint)
    result_dir = tmp_path / "evaluation"
    subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "evaluate.py"),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(result_dir),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["variant"] == "HazeGroupNet-T"
    assert summary["overall"]["num_images"] == 1
