# Installation

## Requirements

The code targets Python 3.10 or newer and PyTorch 2.1 or newer. Install a
PyTorch build that matches the CUDA driver on the target machine before
installing the remaining dependencies.

```bash
git clone https://github.com/jxy0413/HazeGroupNet.git
cd HazeGroupNet
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

The CUDA index above is an example only. Consult the official PyTorch
installation selector for the correct command for your operating system,
Python version, CUDA runtime, or CPU-only setup.

## Optional development dependencies

```bash
pip install -e ".[dev]"
pytest -q
```

## Sanity check

```bash
python tools/profile.py --config configs/rrshid/tiny.yaml --height 256 --width 256
```

This command instantiates a model and reports parameter count and convolutional
MACs. A profiler result can differ across library versions if unsupported
operators are counted differently; use the same profiler and input shape for
fair comparisons.

## Hardware notes

- Training 512x512 images may require gradient accumulation on a single GPU.
  Every released configuration uses an effective batch size of 32; the
  SateHaze1k Large recipe uses a microbatch of 4 with 8 accumulation steps.
- Evaluation and inference accept arbitrary image sizes. The model pads to the
  internal scale factor and crops the restoration back to the input size.
- The paper's latency values were collected separately under the stated
  hardware protocol; they are not bundled as checkpoint benchmarks in this
  repository.
