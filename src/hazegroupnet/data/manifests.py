"""CSV manifests for reproducible paired-image evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REQUIRED_COLUMNS = {"image_id", "hazy_path", "gt_path"}


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    hazy_path: Path
    gt_path: Path
    split: str = ""
    haze_level: str = ""


def _resolve_path(dataset_root: Path, value: str) -> Path:
    if not value:
        raise ValueError("manifest image paths must not be empty")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"manifest image paths must be relative: {value!r}")
    root = dataset_root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes the dataset root: {value!r}") from error
    return resolved


def read_manifest(path: str | Path, dataset_root: str | Path) -> list[ManifestRecord]:
    """Read a CSV manifest and resolve relative paths against ``dataset_root``.

    Required columns are ``image_id,hazy_path,gt_path``. Optional ``split`` and
    ``haze_level`` fields are passed through to the output report.
    """
    path = Path(path)
    dataset_root = Path(dataset_root)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Manifest {path} is missing required columns: {sorted(missing)}")
        records: list[ManifestRecord] = []
        ids: set[str] = set()
        for row_index, row in enumerate(reader, start=2):
            image_id = (row.get("image_id") or "").strip()
            if not image_id:
                raise ValueError(f"Manifest {path}:{row_index} has an empty image_id")
            if image_id in ids:
                raise ValueError(f"Manifest {path} contains duplicate image_id {image_id!r}")
            ids.add(image_id)
            hazy = _resolve_path(dataset_root, (row.get("hazy_path") or "").strip())
            gt = _resolve_path(dataset_root, (row.get("gt_path") or "").strip())
            if not hazy.is_file() or not gt.is_file():
                missing_paths = [str(p) for p in (hazy, gt) if not p.is_file()]
                raise FileNotFoundError(f"Manifest {path}:{row_index} references missing file(s): {missing_paths}")
            records.append(
                ManifestRecord(
                    image_id=image_id,
                    hazy_path=hazy,
                    gt_path=gt,
                    split=(row.get("split") or "").strip(),
                    haze_level=(row.get("haze_level") or "").strip(),
                )
            )
    if not records:
        raise ValueError(f"Manifest {path} contains no records")
    return records


def write_manifest(
    records: Iterable[ManifestRecord], path: str | Path, dataset_root: str | Path
) -> None:
    """Write a portable manifest with paths relative to ``dataset_root`` when possible."""
    path = Path(path)
    root = Path(dataset_root).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    def relative_path(item: Path) -> str:
        try:
            return item.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"manifest path lies outside the dataset root: {item}") from error

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "hazy_path", "gt_path", "split", "haze_level"])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "image_id": record.image_id,
                "hazy_path": relative_path(record.hazy_path),
                "gt_path": relative_path(record.gt_path),
                "split": record.split,
                "haze_level": record.haze_level,
            })
