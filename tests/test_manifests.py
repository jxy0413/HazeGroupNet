from __future__ import annotations

from pathlib import Path

import pytest


def test_manifest_loader_preserves_relative_paths(manifest_file) -> None:
    from hazegroupnet.data import read_manifest

    records = read_manifest(manifest_file, manifest_file.parent)
    assert len(records) == 1
    record = records[0]
    assert record.image_id == "scene_001"
    assert record.hazy_path == manifest_file.parent / "hazy" / "scene_001.png"
    assert record.gt_path == manifest_file.parent / "clean" / "scene_001.png"
    assert record.split == "test"
    assert record.haze_level == "thin"


def test_manifest_loader_rejects_absolute_and_traversal_paths(
    manifest_file: Path,
    tmp_path: Path,
) -> None:
    from hazegroupnet.data import read_manifest

    absolute = (tmp_path / "hazy" / "scene_001.png").resolve()
    manifest_file.write_text(
        "image_id,hazy_path,gt_path\n"
        f"absolute,{absolute.as_posix()},clean/scene_001.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be relative"):
        read_manifest(manifest_file, tmp_path)

    manifest_file.write_text(
        "image_id,hazy_path,gt_path\n"
        "escape,../outside.png,clean/scene_001.png\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes the dataset root"):
        read_manifest(manifest_file, tmp_path)
