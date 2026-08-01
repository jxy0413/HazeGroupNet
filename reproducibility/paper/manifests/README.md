# Manifest specification

The paper uses local, machine-specific manifests to avoid redistributing
dataset paths and image metadata. This directory documents their portable
format and released split sizes only.

## CSV schema

```csv
image_id,hazy_path,gt_path,split,haze_level
rrshid_0001,hazy/0001.png,clear/0001.png,test,moderate
```

| Field | Required | Description |
|:--|:--:|:--|
| `image_id` | yes | Unique, stable identifier. |
| `hazy_path` | yes | Input image path relative to the dataset root. |
| `gt_path` | yes | Clean reference path relative to the dataset root. |
| `split` | yes | One of `train`, `val`, `test`. |
| `haze_level` | no | `thin`, `moderate`, `thick`, or another documented label. |

The public loader rejects absolute paths and path traversal outside the given
dataset root. Use separate files for each split or a single file filtered by
the `split` field.

## Protocol safeguards

- Freeze the test manifest before checkpoint selection.
- Hash manifests and record the hash with each evaluation summary.
- Do not pool RRSHID and SateHaze1k training entries.
- For zero-shot transfer, do not use target-domain validation or test images
  for training, model selection, or normalization fitting.
