"""Conservative discovery and pairing helpers for paired restoration data."""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HAZY_ALIASES = {"hazy", "haze", "foggy", "input", "inputs"}
GT_ALIASES = {
    "gt", "clear", "clean", "groundtruth", "ground_truth", "haze_free",
    "haze-free", "hazefree",
}


def image_files(directory: Path) -> list[Path]:
    """Return supported images recursively in deterministic order."""
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_role_directories(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    """Find non-empty, conventionally named hazy and reference directories."""
    hazy: list[Path] = []
    ground_truth: list[Path] = []
    for directory in [dataset_root, *dataset_root.rglob("*")]:
        if not directory.is_dir():
            continue
        name = directory.name.lower().strip()
        if name in HAZY_ALIASES and image_files(directory):
            hazy.append(directory)
        if name in GT_ALIASES and image_files(directory):
            ground_truth.append(directory)
    return sorted(set(hazy)), sorted(set(ground_truth))


def _normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(^|[_\\- ])(hazy|haze|foggy|gt|clear|clean)(?=$|[_\\- ])", "_", stem)
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def pair_images(
    hazy_dir: Path, gt_dir: Path
) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    """Pair images without guessing when multiple candidate references exist.

    Relative paths are preferred, followed by exact stems and then a conservative
    normalization that removes common ``hazy``/``clear`` tokens.
    """
    hazy_files = image_files(hazy_dir)
    gt_files = image_files(gt_dir)
    gt_by_relative = {
        str(path.relative_to(gt_dir).with_suffix("")).lower(): path for path in gt_files
    }
    gt_by_stem: dict[str, list[Path]] = {}
    gt_by_normalized: dict[str, list[Path]] = {}
    for path in gt_files:
        gt_by_stem.setdefault(path.stem.lower(), []).append(path)
        gt_by_normalized.setdefault(_normalized_stem(path), []).append(path)

    pairs: list[tuple[Path, Path]] = []
    unmatched_hazy: list[Path] = []
    used_gt: set[Path] = set()
    for hazy_path in hazy_files:
        relative_key = str(hazy_path.relative_to(hazy_dir).with_suffix("")).lower()
        candidate = gt_by_relative.get(relative_key)
        if candidate in used_gt:
            candidate = None
        if candidate is None:
            exact = [p for p in gt_by_stem.get(hazy_path.stem.lower(), []) if p not in used_gt]
            if len(exact) == 1:
                candidate = exact[0]
        if candidate is None:
            normalized = [
                p for p in gt_by_normalized.get(_normalized_stem(hazy_path), []) if p not in used_gt
            ]
            if len(normalized) == 1:
                candidate = normalized[0]
        if candidate is None:
            unmatched_hazy.append(hazy_path)
            continue
        pairs.append((hazy_path, candidate))
        used_gt.add(candidate)
    return pairs, unmatched_hazy, [path for path in gt_files if path not in used_gt]
