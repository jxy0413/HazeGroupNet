"""Train HazeGroupNet from a paired-image CSV manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hazegroupnet.data import PairedImageDataset, read_manifest  # noqa: E402
from hazegroupnet.metrics import compute_metrics  # noqa: E402
from hazegroupnet.utils.config import (  # noqa: E402
    create_model_from_config,
    load_config,
)
from hazegroupnet.utils.early_stopping import (  # noqa: E402
    EarlyStoppingConfig,
    EarlyStoppingState,
)
from hazegroupnet.utils.reproducibility import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration/manifests and print the run plan.",
    )
    return parser.parse_args()


def fft_l1(prediction: Tensor, target: Tensor) -> Tensor:
    """L1 distance between real/imaginary rFFT2 coefficients."""

    prediction_fft = torch.view_as_real(torch.fft.rfft2(prediction.float()))
    target_fft = torch.view_as_real(torch.fft.rfft2(target.float()))
    return F.l1_loss(prediction_fft, target_fft)


def tensor_to_rgb(tensor: Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: PairedImageDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
) -> DataLoader[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch in loader:
            hazy = batch["hazy"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prediction = model(hazy)
            for predicted_image, target_image in zip(prediction, target):
                rows.append(
                    compute_metrics(
                        tensor_to_rgb(predicted_image),
                        tensor_to_rgb(target_image),
                    )
                )
    model.train()
    return {
        key: float(np.mean([row[key] for row in rows])) for key in ("psnr", "ssim", "delta_e00")
    }


def atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_grad_scaler(use_amp: bool) -> Any:
    """Create a scaler across supported PyTorch 2.x AMP APIs."""

    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    step: int,
    best: dict[str, float | int],
    config: dict[str, Any],
    generator: torch.Generator,
    early_stopping_state: EarlyStoppingState | None,
    manifest_sha256: dict[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "format": "hazegroupnet_public_training_v1",
        "step": step,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best": best,
        "config": config,
        "manifest_sha256": manifest_sha256,
        "data_generator_state": generator.get_state(),
    }
    if early_stopping_state is not None:
        payload["early_stopping_state"] = early_stopping_state.to_dict()
    return payload


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")

    config = load_config(args.config)
    experiment = config["experiment"]
    data_config = config["data"]
    train_config = config["train"]

    train_records = read_manifest(args.train_manifest, args.dataset_root)
    val_records = read_manifest(args.val_manifest, args.dataset_root)
    manifest_sha256 = {
        "train": sha256_file(args.train_manifest),
        "validation": sha256_file(args.val_manifest),
    }
    seed = int(experiment.get("seed", 2026))
    crop_size = int(data_config["crop_size"])
    microbatch_size = int(train_config["microbatch_size"])
    accumulation_steps = int(train_config["accumulation_steps"])
    max_steps = int(train_config["max_steps"])
    validation_interval = int(train_config["validation_interval"])
    save_interval = int(train_config["save_interval"])
    learning_rate = float(train_config["learning_rate"])
    minimum_learning_rate = float(train_config.get("minimum_learning_rate", 4e-8))
    weight_decay = float(train_config.get("weight_decay", 0.0))
    fft_weight = float(train_config.get("fft_weight", 0.1))
    early_stopping_config = None
    early_stopping_state = None
    if "early_stopping" in train_config:
        early_stopping_config = EarlyStoppingConfig.from_mapping(
            train_config["early_stopping"]
        )
        early_stopping_config.validate(
            max_steps=max_steps,
            validation_interval=validation_interval,
        )
        early_stopping_state = EarlyStoppingState()

    if (
        min(
            microbatch_size,
            accumulation_steps,
            max_steps,
            validation_interval,
            save_interval,
        )
        <= 0
    ):
        raise ValueError("training counts and intervals must be positive")

    plan = {
        "experiment": experiment.get("name", args.config.stem),
        "variant": config["model"]["variant"],
        "train_images": len(train_records),
        "validation_images": len(val_records),
        "crop_size": crop_size,
        "max_steps": max_steps,
        "microbatch_size": microbatch_size,
        "accumulation_steps": accumulation_steps,
        "effective_batch_size": microbatch_size * accumulation_steps,
        "learning_rate": learning_rate,
        "minimum_learning_rate": minimum_learning_rate,
        "weight_decay": weight_decay,
        "fft_weight": fft_weight,
        "seed": seed,
        "manifest_sha256": manifest_sha256,
        "early_stopping": (
            {
                "eligibility_step": early_stopping_config.eligibility_step,
                "earliest_stop_step": early_stopping_config.earliest_stop_step,
                "patience": early_stopping_config.patience,
                "min_delta_psnr": early_stopping_config.min_delta_psnr,
                "min_delta_ssim": early_stopping_config.min_delta_ssim,
                "psnr_guard": early_stopping_config.psnr_guard,
            }
            if early_stopping_config is not None
            else None
        ),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataset = PairedImageDataset(
        train_records,
        crop_size=crop_size,
        augment=True,
    )
    val_dataset = PairedImageDataset(
        val_records,
        crop_size=None,
        augment=False,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        generator=generator,
    )

    model = create_model_from_config(config).to(device)
    optimizer_name = str(train_config.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError("the public training entry point currently supports AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    scheduler_name = str(train_config.get("scheduler", "cosine")).lower()
    if scheduler_name != "cosine":
        raise ValueError("the public entry point currently supports cosine scheduling")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=minimum_learning_rate,
    )
    use_amp = device.type == "cuda" and bool(train_config.get("amp", True)) and not args.no_amp
    scaler = create_grad_scaler(use_amp)

    start_step = 0
    best: dict[str, float | int] = {
        "step": 0,
        "psnr": float("-inf"),
        "ssim": float("-inf"),
        "delta_e00": float("inf"),
    }
    history: list[dict[str, float | int]] = []
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("resume checkpoint is not a public training checkpoint")
        if payload.get("config") != config:
            raise ValueError("resume checkpoint configuration does not match --config")
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError("resume checkpoint manifests do not match the supplied manifests")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload.get("scaler", {}))
        start_step = int(payload["step"])
        best = dict(payload.get("best", best))
        if "data_generator_state" in payload:
            generator.set_state(payload["data_generator_state"])
        if early_stopping_state is not None and "early_stopping_state" in payload:
            early_stopping_state = EarlyStoppingState.from_mapping(
                payload["early_stopping_state"]
            )
        history_path = args.output_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    save_json(plan, args.output_dir / "run_plan.json")

    model.train()
    iterator = iter(train_loader)
    stop_requested = False
    for step in range(start_step + 1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_pixel = 0.0
        accumulated_fft = 0.0

        for _ in range(accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            hazy = batch["hazy"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                prediction = model(hazy)
                pixel_loss = F.l1_loss(prediction.float(), target.float())
                frequency_loss = fft_l1(prediction, target)
                loss = pixel_loss + fft_weight * frequency_loss
                scaled_loss = loss / accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach()) / accumulation_steps
            accumulated_pixel += float(pixel_loss.detach()) / accumulation_steps
            accumulated_fft += float(frequency_loss.detach()) / accumulation_steps

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step}/{max_steps} loss={accumulated_loss:.6f} "
                f"pixel={accumulated_pixel:.6f} fft={accumulated_fft:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f}"
            )

        should_validate = step % validation_interval == 0 or step == max_steps
        if should_validate:
            metrics = validate(model, val_loader, device)
            row: dict[str, float | int] = {"step": step, **metrics}
            history.append(row)
            save_json(history, args.output_dir / "history.json")
            early_stopping_decision = None
            if early_stopping_config is not None and early_stopping_state is not None:
                early_stopping_decision = early_stopping_state.observe(
                    step=step,
                    psnr=metrics["psnr"],
                    ssim=metrics["ssim"],
                    config=early_stopping_config,
                )
                stop_requested = bool(early_stopping_decision["should_stop"])
            improved = metrics["psnr"] > float(best["psnr"]) or (
                metrics["psnr"] == float(best["psnr"]) and metrics["ssim"] > float(best["ssim"])
            )
            if improved:
                best = {"step": step, **metrics}
                atomic_torch_save(
                    checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=step,
                        best=best,
                        config=config,
                        generator=generator,
                        early_stopping_state=early_stopping_state,
                        manifest_sha256=manifest_sha256,
                    ),
                    args.output_dir / "best.pt",
                )
                save_json(best, args.output_dir / "best.json")
            print(
                json.dumps(
                    {
                        "validation": row,
                        "best": best,
                        "early_stopping": early_stopping_decision,
                    },
                    indent=2,
                )
            )

        should_save = step % save_interval == 0 or step == max_steps or stop_requested
        if should_save:
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    best=best,
                    config=config,
                    generator=generator,
                    early_stopping_state=early_stopping_state,
                    manifest_sha256=manifest_sha256,
                ),
                args.output_dir / "last.pt",
            )
        if stop_requested:
            save_json(
                {
                    "status": "early_stopped",
                    "step": step,
                    "best": best,
                    "reason": "frozen_validation_plateau",
                },
                args.output_dir / "status.json",
            )
            break

    final_status = "early_stopped" if stop_requested else "complete"
    if not stop_requested:
        save_json(
            {"status": final_status, "step": max_steps, "best": best},
            args.output_dir / "status.json",
        )
    print(json.dumps({"status": final_status, "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
