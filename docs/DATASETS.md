# Datasets and manifests

## Non-redistribution policy

This repository contains no images, labels, dataset archives, or checkpoints.
Obtain RRSHID, SateHaze1k, RICE, and any other benchmark only from its
original provider and comply with that provider's license and access terms.
In particular, this code release does not grant rights to redistribute or use
any dataset commercially.

## Paired-image layout

The generic tools use a dataset root plus a CSV manifest. Image paths in the
manifest must be relative to the supplied dataset root; absolute paths are
rejected by the public manifest loader.

```text
dataset_root/
├── hazy/
│   ├── scene_0001.png
│   └── scene_0002.png
├── clean/
│   ├── scene_0001.png
│   └── scene_0002.png
└── manifests/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Minimal manifest columns are:

| Column | Meaning |
|:--|:--|
| `image_id` | Stable identifier used in reports and saved predictions. |
| `hazy_path` | Relative path to the degraded input. |
| `gt_path` | Relative path to the paired clean reference. |
| `split` | Optional label such as `train`, `val`, or `test`. |
| `haze_level` | Optional category such as `thin`, `moderate`, or `thick`. |

Example:

```csv
image_id,hazy_path,gt_path,split,haze_level
scene_0001,hazy/scene_0001.png,clean/scene_0001.png,test,thin
```

## Paper protocols

The paper uses separate in-domain protocols for RRSHID and SateHaze1k, and
uses RICE only for zero-shot cross-degradation evaluation. Do not pool their
training sets. The release provides manifest specifications, not the
underlying manifest entries, because paths and redistribution permissions are
dataset-provider dependent. See
`reproducibility/paper/manifests/README.md`.

For reported paper values, retain the released test sets unchanged, do not use
test samples for checkpoint selection, and record the exact dataset release
used. RICE evaluation should be described as zero-shot robustness under an
unseen cloud-like degradation, not as supervised cloud-removal training.
