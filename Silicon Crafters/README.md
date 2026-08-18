# Silicon Crafters — KLA Problem Statement Submission

**AI-Based Restoration of Degraded Images for Semiconductor Inspection**

---

## Team

**Team Name:** Silicon Crafters

---

## Project Overview

This submission implements **AMSR-Net** (Adaptive Multi-Expert Semiconductor Restoration Network), a hybrid deep learning model that performs **simultaneous denoising and 2× super-resolution** on degraded semiconductor SEM inspection images.

- **Input:** Degraded, noisy low-resolution images — 128×128 grayscale `.npy` arrays
- **Output:** Restored high-resolution images — 256×256 grayscale `.npy` arrays, values in `[0, 1]`

---

## Submission Structure

```
Silicon Crafters/
├── run.py               ← Entry point
├── requirements.txt     ← Python dependencies (pinned)
├── README.md            ← This file
└── models/
    ├── amsr_net_best.pth       ← Trained model weights (~12 MB)
    ├── config.py               ← Model hyperparameters
    ├── amsr_net.py             ← AMSR-Net architecture
    ├── baseline_cnn.py         ← CNN residual block components
    ├── restormer_block.py      ← Restormer MDTA attention blocks
    ├── swin_block.py           ← Swin Transformer window attention blocks
    ├── __init__.py
    ├── dataset/
    │   ├── semiconductor_dataset.py   ← Normalisation utilities
    │   └── __init__.py
    └── metrics/
        ├── psnr.py
        ├── ssim_metric.py
        └── __init__.py
```

---

## Setup

> **Prerequisites:** Python 3.10+, pip, NVIDIA GPU with CUDA 11.8+ (recommended)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**For NVIDIA GPU support** (recommended — significantly faster):

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 Pillow==10.3.0 scipy==1.13.0 tqdm==4.66.4
```

> The script automatically detects and uses any available NVIDIA GPU. If no GPU is found, it falls back to CPU without any configuration change.

---

## Running the Solution

```bash
python run.py <input-dir> <output-dir>
```

| Argument | Description |
|---|---|
| `<input-dir>` | Directory containing degraded `.npy` images (NoisyLR) |
| `<output-dir>` | Directory where restored `.npy` images will be saved (created automatically) |

### Example

```bash
python run.py ./dataset/NoisyLR ./outputs/restored
```

### What happens

1. All `.npy` files in `<input-dir>` are discovered automatically.
2. `<output-dir>` is created if it does not exist.
3. Each image is restored using AMSR-Net inference.
4. One `.npy` file is saved to `<output-dir>` for every input, with the **same filename**.
5. A summary is printed to console on completion.

---

## Output Format

| Property | Value |
|---|---|
| Format | NumPy `.npy` file |
| dtype | `float32` |
| Shape | `(256, 256)` — i.e., `(H, W)` |
| Value range | `[0.0, 1.0]` |
| NaN / Inf | None (validated before saving) |

---

## Model Architecture — AMSR-Net

```
Input NoisyLR (128×128)
       │
Stage 0: 3×3 Shallow Feature Extraction Conv
       │
Stage 1: CNN Residual Blocks × 4  (local texture extraction)
       │
Stage 2: Restormer MDTA Blocks × 6  (global channel attention / denoising)
       │
Stage 3: Swin Transformer Pairs × 2  (local spatial attention / SR prep)
       │
Stage 4: Feature Fusion (1×1 Conv + global residual)
       │
Stage 5: 2× PixelShuffle Upsampler (128→256)
       │
Stage 6: Reconstruction Head + Bilinear Residual Skip
       │
Output Restored HR (256×256)
```

**Parameters:** ~0.97M  |  **Inference time:** ~49 ms/image (RTX 2050)

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=100, η_min=1e-7) |
| Loss | 0.5×Charbonnier + 0.2×SSIM + 0.1×Sobel Edge |
| Precision | Mixed precision (torch.amp.autocast) |
| Hardware | NVIDIA RTX 2050 (4.3 GB VRAM) |
| Epochs | 100 |

---

## Results

| Model | PSNR (dB) | SSIM |
|---|:---:|:---:|
| Bicubic Baseline | 25.79 | 0.647 |
| **AMSR-Net (Ours)** | **29.25** | **0.795** |

---

## Notes

- The solution runs **fully offline** — no internet access, API keys, or additional model downloads are required.
- All model weights are bundled in `models/amsr_net_best.pth`.
- No user interaction is required during inference.
- Input arrays may have values slightly outside `[0, 1]` due to noise — the preprocessing pipeline clips and normalises them correctly.
