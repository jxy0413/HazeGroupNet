# Saved-image 8-bit evaluation audit

Audit date: 2026-08-02

## Frozen convention

The manuscript reports PSNR, SSIM, and CIEDE2000 from saved-image-precision
8-bit sRGB predictions. The reported paper values were generated with the
paper's frozen 8-bit evaluator and are therefore retained unchanged.

The public evaluator now implements the same convention with one shared data
path:

1. clip the model output to `[0, 1]`;
2. multiply by 255 and round to the nearest integer;
3. encode the result as an `HxWx3` `uint8` RGB array;
4. compute every metric from that quantized array;
5. when requested, save the same array as a PNG without another conversion.

Metric computation does not depend on whether `--save-predictions` is enabled.

## Automated consistency check

`tests/test_cli_smoke.py` evaluates a checkpoint with prediction saving
enabled, reloads the written PNG, recomputes all metrics from the reloaded
8-bit array, and verifies equality with `per_image.csv` to numerical
precision. This guards against future divergence between the evaluated image
and the saved prediction.
