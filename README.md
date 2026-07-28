# InterpolateNautilus — Multi-Model Video Frame Interpolation

[![arXiv](https://img.shields.io/badge/arXiv-2512.18241-b31b1b.svg)](https://arxiv.org/abs/2512.18241)
[![License](https://img.shields.io/github/license/ben813/sg-rife)](LICENSE)

Unified CLI for video frame interpolation supporting multiple VFI models.

Current models:
- **rife** — Base RIFE (Real-Time Intermediate Flow Estimation), v4.x. Usable but produces noticeably inferior output — ghosting, blurring, and temporal artifacts are common - call this dog shit is unfair but comparatively - it is not good.
- **sg-rife** — SG-RIFE with DINOv3 semantic injection (RIFE + frozen DINOv3 ViT-S/16). This is the model you actually want. Originally developed by [Wong, Wu, and Lu](https://arxiv.org/abs/2512.18241) — not me.

---

## Installation

```bash
git clone --recursive <repo-url>
cd InterpolateNautilus
python -m venv .venv

# PyTorch with CUDA 13
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130 --python .venv/bin/python

uv pip install -r requirements.txt --python .venv/bin/python
uv pip install -e vfi_models/sg_rife/dinov3_repo --python .venv/bin/python

# SageAttention 2.2.0 (quantized attention kernels)
cd SageAttention && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 ../.venv/bin/python setup.py install
cd ..
```

### Checkpoints

Place model weights in `checkpoints/`:

```
checkpoints/
├── rife/
│   └── rife49.pth
└── sg_rife/
    ├── flownet.pkl
    └── dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

---

## Usage

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

# SageAttention (SG-RIFE only — ~2x attention speedup)
.venv/bin/python interpolate.py --model sg-rife --video input.mp4 --exp=2 --with-sage
```

### Flags

| Flag | Default | Description |
|:---|:---|:---|
| `--model` | `rife` | Model: `rife` or `sg-rife` |
| `--img` | — | Two input image paths |
| `--video` | — | Input video file |
| `--output` | auto | Output video path |
| `--exp` | `4` | Interpolation factor (2^exp frames per pair) |
| `--ratio` | `0` | Single frame at time 0.0–1.0 |
| `--scale` | `1.0` | Resolution scale (0.25, 0.5, 1.0, 2.0, 4.0) |
| `--fps` | auto | Output FPS |
| `--fp16` | off | Half-precision inference |
| `--png` | off | Output PNG sequence instead of video |
| `--ext` | `mp4` | Output video container |
| `--with-sage` | off | SageAttention acceleration (SG-RIFE only) |
| `--model-dir` | auto | Custom checkpoint directory |

---

## Architecture

### SG-RIFE

SG-RIFE bridges the gap between the high throughput of flow-based interpolation (RIFE) and the superior perceptual quality of diffusion-based models by injecting dense semantic priors from **DINOv3** through **Deformable Semantic Fusion (DSF)** modules. Only ~16% of parameters are trained.

| Base RIFE | SG-RIFE |
|:---:|:---:|
| ![RIFE](https://raw.githubusercontent.com/ben813/sg-rife/main/demo/rife.gif) | ![SG-RIFE](https://raw.githubusercontent.com/ben813/sg-rife/main/demo/sg-rife.gif) |

**Paper:** [arXiv:2512.18241](https://arxiv.org/abs/2512.18241)

---

## Project Structure

```
InterpolateNautilus/
├── interpolate.py              # Unified CLI
├── vfi_models/
│   ├── rife/rife_arch.py       # Base RIFE (1 file)
│   └── sg_rife/                # SG-RIFE
│       ├── sg_rife_arch.py     # Architecture + model wrapper (1 file)
│       ├── dino_config.py      # DINOv3 paths & hyperparams
│       ├── dino_modules/       # DINOv3 adapters
│       └── dinov3_repo/        # Git submodule
├── checkpoints/                # Model weights
├── pytorch_msssim/             # SSIM metric
└── test/                       # Test assets
```

---

## Citation

```bibtex
@article{wong2025sg,
  title={SG-RIFE: Semantic-Guided Real-Time Intermediate Flow Estimation with Diffusion-Competitive Perceptual Quality},
  author={Wong, Pan Ben and Wu, Chengli and Lu, Hanyue},
  journal={arXiv preprint arXiv:2512.18241},
  year={2025}
}
```

## Acknowledgements

- **RIFE**: [hzwer/ECCV2022-RIFE](https://github.com/hzwer/ECCV2022-RIFE)
- **DINOv3**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **Dino U-Net**: [yifangao112/DinoUNet](https://github.com/yifangao112/DinoUNet)