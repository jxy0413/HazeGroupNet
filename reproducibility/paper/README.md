# Paper reproducibility metadata

This directory records public metadata needed to interpret the experimental
protocol in the HazeGroupNet manuscript. It is not a checkpoint release.

## Scope

- `expected_metrics.json` records the principal **reported paper results**.
- `EVALUATION_8BIT_AUDIT.md` records the saved-image 8-bit metric convention
  and its automated PNG read-back consistency check.
- `manifests/` documents the required manifest schema without redistributing
  dataset paths or image identifiers.
- `frozen_protocols/` describes fixed evaluation and profiling conventions.

The source code exposes generic CLIs. To reproduce a new run, provide your own
officially obtained data, manifests, checkpoint-selection procedure, and
checkpoint. Do not select a checkpoint against a test set.

## Reported configurations

The paper evaluates Tiny, Small, and Large variants. The Large variant is the
default accuracy configuration in the manuscript; no checkpoint is bundled for
any variant.

## Dataset use

RRSHID and SateHaze1k are used in separate in-domain protocols. RICE1 and
RICE2 are only used for zero-shot cross-degradation evaluation. Dataset files
and official manifests are not republished here.
