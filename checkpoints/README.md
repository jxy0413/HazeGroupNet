# Representative checkpoints

This directory hosts a six-checkpoint release matrix: one validation-selected
HazeGroupNet-T/S/L checkpoint for RRSHID and one for SateHaze1k. The files are
representative trained models for inference and checkpoint-specific
verification. They are not an ensemble and do not reproduce the manuscript's
three-seed mean and sample standard deviation by themselves.

## Current inventory

| Dataset | HGN-T | HGN-S | HGN-L |
|---|---|---|---|
| RRSHID | pending | seed 2026, step 38,000 | training: seed 2027 |
| SateHaze1k | pending | pending | seed 2027, step 11,000 |

The machine-readable source of truth is [`MANIFEST.json`](MANIFEST.json).
Every public file is a CPU FP32 PyTorch `state_dict` without optimizer,
scheduler, AMP-scaler, RNG, dataset image, or private-path state. Load it with
`torch.load(path, map_location="cpu", weights_only=True)` and use
`model.load_state_dict(state_dict, strict=True)`.

## Checkpoint-specific verification

- `RRSHID_HGN-S_best.pt`: 304-image saved-8-bit replay, PSNR 24.645608,
  SSIM 0.706905, and CIEDE2000 5.287867.
- `SateHaze1k_HGN-L_best.pt`: 135-image saved-8-bit replay, PSNR 22.932620,
  SSIM 0.851918, and CIEDE2000 7.173405.

These values describe the public files themselves. Consult the manuscript and
`reproducibility/paper/` for the separately reported multi-seed study.

## Usage conditions

No benchmark images are redistributed. Obtain RRSHID and SateHaze1k from
their original providers and comply with their terms. See
[`WEIGHTS_NOTICE.md`](WEIGHTS_NOTICE.md) and the repository-level
[`NOTICE.md`](../NOTICE.md).
