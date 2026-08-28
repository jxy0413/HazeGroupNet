# Reproducibility guide

## What this release supports

The repository provides a self-contained implementation of HazeGroupNet,
model configurations, generic paired-image CLIs, metric code, public protocol
specifications, and a curated representative-checkpoint set. It does not
provide benchmark image data.

## Paper versus local reruns

The values in the manuscript are **reported paper results** obtained with the
paper's frozen data splits, checkpoint-selection rules, evaluator, and
hardware protocol. A six-checkpoint matrix is being populated with one
representative validation-selected run per dataset and model scale; its
current inventory is recorded in `checkpoints/MANIFEST.json`. It is intended
for direct inference and checkpoint-specific verification and does not replace
the 18 independent runs underlying the manuscript's three-seed mean and sample
SD.

For an independent rerun:

1. Obtain the dataset from its official source and document its release.
2. Create disjoint train/validation/test manifests using the supplied schema.
3. Train with one of `configs/<dataset>/<variant>.yaml`.
4. Select a checkpoint using validation data only.
5. Run `tools/evaluate.py` exactly once on the frozen test manifest.
6. Archive the manifest, configuration, checkpoint hash, per-image CSV, and
   summary JSON alongside the result.

The public `--resume` path verifies the complete configuration and the SHA-256
hashes of both manifests before restoring optimization state. It is intended
for safe continuation, but it does not claim bitwise-identical replay of a
partially consumed multi-worker data-loader epoch.

## Evaluation conventions

Unless a new experiment explicitly states otherwise:

- Clip predictions to `[0, 1]`, round them to the nearest 8-bit sRGB value,
  and evaluate the resulting `uint8` RGB array. When predictions are saved,
  the evaluator writes this exact array to PNG; metrics never use the
  pre-quantized floating-point output.
- Compute PSNR from per-image RGB mean-squared error and average the
  per-image scores.
- Compute SSIM with an 11x11 Gaussian window (sigma 1.5), population
  covariance, and the valid filtered interior.
- Compute CIEDE2000 per RGB-to-Lab pixel and average first within each image,
  then over the evaluation set.
- Report full-test-set metrics directly rather than averaging subset means.
- Keep thin, moderate, and thick subset scores separate when the dataset
  provides those labels.
- Use identical output resolution for all compared predictions.
- For zero-shot transfer, do not train, fine-tune, or select checkpoints on
  the target-domain validation or test samples.

## Training conventions

The released RRSHID and SateHaze1k recipes use AdamW with
`betas=(0.9, 0.999)`, zero weight decay, an initial learning rate of
`4e-4`, and cosine annealing to `4e-8`. The effective batch size is 32.
The restoration objective is pixel-domain L1 plus `0.1` times the L1
distance between the real and imaginary components of the unnormalized
two-dimensional real FFT. Paired horizontal flips and rotations by multiples
of 90 degrees are applied identically to the hazy and reference images.

Quality-based early stopping is frozen in each released configuration.
RRSHID establishes its monitoring anchor at step 40,000 and can first stop at
step 50,000; SateHaze1k establishes its anchor at step 10,000 and can first
stop at step 12,500. Both use five consecutive validation points without a
meaningful improvement. A meaningful improvement is either at least
0.03 dB PSNR, or at least 0.0005 SSIM while remaining within 0.03 dB of the
best PSNR reference. Best-checkpoint selection remains separate and uses
validation PSNR with SSIM as the tie-breaker.

## Frozen-protocol metadata

The paper-specific protocol descriptions and reported numbers are stored under
`reproducibility/paper/`. Checkpoint-specific hashes and availability are
stored under `checkpoints/`.
