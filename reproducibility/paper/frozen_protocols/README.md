# Frozen protocol summary

## In-domain evaluation

RRSHID and SateHaze1k use separate train/validation/test protocols. The test
manifest is frozen before model selection. Overall PSNR, SSIM, and CIEDE2000
are calculated directly over all test images; haze-density subsets are also
reported when labels are available.

## Zero-shot protocols

For RRSHID-to-SateHaze1k and SateHaze1k-to-RRSHID transfer, source-domain
training is retained unchanged. Target-domain data are not used for training,
fine-tuning, or checkpoint selection. RICE1/RICE2 results are described as
unseen cloud-like degradation robustness rather than supervised cloud removal.

## Profiling

Paper complexity counts are convolutional MACs at 256x256 input. Hardware
latency and peak-memory measurements use the full model, including haze-proxy
computation. Reproduce a fair comparison with one hardware/software stack,
batch size 1, warm-up iterations, synchronized timing, and a documented
precision mode.

## Checkpoint provenance

No checkpoint, private log, or immutable internal manifest is released. The
expected-metrics file is a manuscript record, not a substitute for a model
artifact.
