from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    for role in ("hazy", "clean"):
        directory = tmp_path / role
        directory.mkdir()
        Image.new("RGB", (2, 2), color=(128, 128, 128)).save(directory / "scene_001.png")

    path = tmp_path / "manifest.csv"
    path.write_text(
        "image_id,hazy_path,gt_path,split,haze_level\n"
        "scene_001,hazy/scene_001.png,clean/scene_001.png,test,thin\n",
        encoding="utf-8",
    )
    return path
