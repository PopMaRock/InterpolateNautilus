# AGENTS.md — InterpolateNautilus Developer Guide

This document describes the project structure, coding conventions, and development workflows for InterpolateNautilus. It is intended for both human contributors and AI coding assistants.

---

## Project Overview

InterpolateNautilus is a multi-model video frame interpolation system. It provides a single unified CLI (`interpolate.py`) that supports multiple VFI (Video Frame Interpolation) models selectable via `--model <name>`.

Currently supported models:
- **rife** — Base RIFE (Real-Time Intermediate Flow Estimation), v4.x architecture
- **sg-rife** — SG-RIFE with DINOv3 semantic injection (RIFE + frozen DINOv3 ViT-S/16 + trainable adapters)

Inference only. No training, no benchmarks, no Docker, no notebooks.

**Paper (SG-RIFE):** [arXiv:2512.18241](https://arxiv.org/abs/2512.18241)

---

## Directory Structure

```
InterpolateNautilus/
├── interpolate.py                     # [ENTRY POINT] Unified CLI (--model, --img/--video)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── .gitmodules                        # DINOv3 submodule reference
├── LICENSE                            # License file
│
├── vfi_models/                        # VFI model plugins
│   ├── rife/
│   │   └── rife_arch.py               # Base RIFE architecture (single file)
│   └── sg_rife/
│       ├── sg_rife_arch.py            # SG-RIFE architecture (single file)
│       ├── dino_config.py             # DINOv3 integration config & paths
│       ├── dino_modules/              # DINOv3 adapter modules
│       │   ├── __init__.py
│       │   ├── dino_wrapper.py        # DINOv3 feature extractor (torch.hub local load)
│       │   ├── dino_adapter.py        # Split-FAPM: compressor + refiner
│       │   └── dino_fusion.py         # Deformable Semantic Fusion (DSF) with DCNv2
│       └── dinov3_repo/               # [GIT SUBMODULE] facebookresearch/dinov3
│
├── checkpoints/                       # Model weights
│   ├── rife/
│   │   └── rife49.pth                 # Base RIFE checkpoint
│   └── sg_rife/
│       ├── flownet.pkl                # SG-RIFE trained weights
│       └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth  # DINOv3 backbone
│
├── pytorch_msssim/                    # SSIM metric (PyTorch port)
│   └── __init__.py
│
└── test/                              # Test assets (images/video for validation)
```

---

## Architecture & Data Flow

### SG-RIFE Inference Pipeline

```
Input: img0, img1
    │
    ├──► DinoWrapper.get_features(img0) ──► feats0 (List[Tensor])
    │                                        frozen DINOv3 ViT-S/16
    ├──► DinoWrapper.get_features(img1) ──► feats1 (List[Tensor])
    │
    └──► IFNet(concat(img0, img1), dino_feats=(feats0, feats1))
              │
              ├──► FAPM_Encoder (compress 384D → 256D, FiLM modulation)
              ├──► Warp features via optical flow
              ├──► FAPM_Refiner (SqueezeExcitation + depthwise conv)
              ├──► DSF modules (DCNv2 soft-alignment) at multiple scales
              └──► Output: merged[2] — interpolated frame
```

### Key Design Decisions

1. **BGR vs RGB handling**: OpenCV loads images as BGR. DINOv3 expects RGB. `DinoWrapper.get_features()` flips channels via `img.flip(1)` and applies ImageNet normalization.
2. **Padding**: Images are padded to multiples of 32 (base RIFE) or 64 (HD variants) before processing. Output is cropped back to original dimensions.
3. **Multi-scale inference**: Flow is estimated at scales [4, 2, 1] by default for SG-RIFE, [16, 8, 4, 2, 1] for base RIFE.
4. **Local torch.hub**: DINOv3 is loaded via `torch.hub.load(repo_dir, model_name, source="local")` — it is NOT installed as a pip package. The `dinov3_repo/` submodule must be present.

---

## Coding Standards

### Language & Runtime
- **Python 3.12+**
- **PyTorch 2.13+** with **CUDA 13** (cu130) as the primary deep learning framework
- NVIDIA GPU required (tested on RTX 4080, sm89 / Ada Lovelace)
- CPU fallback is not supported

### Style Conventions
- Indentation: 4 spaces throughout.
- Single quotes for most code, double quotes for docstrings.
- No formatter configured — follow the prevailing style in the file being edited.

### Naming Conventions
- **Classes**: PascalCase (`IFNet`, `RifeModel`, `FAPM_Encoder`)
- **Functions/Methods**: snake_case (`get_features`, `load_model`, `make_inference`)
- **Variables**: snake_case (`img0`, `dino_feats`)

### File Organization
- Model architecture files in `vfi_models/<model>/`
- Checkpoint weights in `checkpoints/<model>/`
- One architecture file per VFI model
- CLI entrypoint at project root

---

## Build, Run, Test

### Package Management

This project uses **uv** for package management. All dependencies are installed into the project-local `.venv/` virtual environment.

PyTorch must be installed with CUDA 13 (cu130) support:
```bash
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130 --python .venv/bin/python
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e vfi_models/sg_rife/dinov3_repo --python .venv/bin/python
```

### SageAttention Installation

SageAttention 2.2.0 is built from source (requires CUDA 13 toolkit available as system `nvcc`):
```bash
cd SageAttention && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 ../.venv/bin/python setup.py install
```

Pre-built wheels are not used — the system CUDA 13.3 nvcc and PyTorch 2.13.0+cu130 must match at build time.

Always use `.venv/bin/python` to run scripts:
```bash
.venv/bin/python interpolate.py --model rife --img img0.png img1.png --exp=4
```

### Setup (from scratch)
```bash
git clone --recursive <repo-url>
cd InterpolateNautilus
python -m venv .venv
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130 --python .venv/bin/python
uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e vfi_models/sg_rife/dinov3_repo --python .venv/bin/python
cd SageAttention && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 ../.venv/bin/python setup.py install
```

### Run Inference

```bash
# Image interpolation
.venv/bin/python interpolate.py --model rife --img a.png b.png --exp=4
.venv/bin/python interpolate.py --model sg-rife --img a.png b.png --exp=4

# Single ratio frame
.venv/bin/python interpolate.py --model rife --img a.png b.png --ratio=0.5

# Video interpolation
.venv/bin/python interpolate.py --model rife --video input.mp4 --exp=2
.venv/bin/python interpolate.py --model sg-rife --video input.mp4 --exp=1 --scale 0.5

# With FP16
.venv/bin/python interpolate.py --model rife --video input.mp4 --exp=2 --fp16
```

### Testing
There is no unit test suite. Run inference on the test assets:
```bash
.venv/bin/python interpolate.py --model rife --img test/I2_0.png test/I2_1.png --exp=2
.venv/bin/python interpolate.py --model sg-rife --img test/I2_0.png test/I2_1.png --exp=2
```
Verify output images appear in `output/`.

---

## Dependencies

### Core Runtime
| Package | Role |
|:---|:---|
| `torch` (PyTorch 2.13+) | Deep learning framework (CUDA 13 / cu130) |
| `torchvision` | Transforms (ImageNet normalization) |
| `triton` | GPU kernel language |
| `sageattention` (2.2.0) | Quantized attention kernels (built from source) |
| `opencv-python` | Image I/O, video capture |
| `numpy` | Array operations |
| `tqdm` | Progress bars |
| `ffmpeg` (system) | Audio extraction/merging for video output |

### DINOv3 (submodule dependencies)
| Package | Role |
|:---|:---|
| `omegaconf` | YAML config for DINOv3 |
| `ftfy` | Text fixing utility |
| `regex` | Advanced regex |
| `scikit-learn` | ML utilities |
| `submitit` | Cluster job submission |
| `termcolor` | Terminal colors |
| `torchmetrics` | Evaluation metrics |

---

## Common Pitfalls

### DINOv3 submodule is not cloned
The `dinov3_repo/` directory is a git submodule. Without it, `torch.hub.load()` will fail.
```bash
git submodule update --init --recursive
```

### CUDA version mismatch when building SageAttention
PyTorch must be installed with CUDA 13 (cu130) to match the system nvcc.
```bash
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130 --python .venv/bin/python
```
Verify with: `.venv/bin/python -c "import torch; print(torch.version.cuda)"` — should output `13.0`.

### Image channel order
- OpenCV (`cv2.imread`) returns **BGR**.
- DINOv3 expects **RGB** with ImageNet normalization.
- `DinoWrapper.get_features()` handles the BGR→RGB flip internally via `img.flip(1)`.
- The base RIFE script does NOT flip channels — it works in BGR natively.

### Padding requirements
Images are padded to multiples of 32 (base RIFE) or 64 (HD variants) before processing. Output is cropped back to `[:h, :w]`.