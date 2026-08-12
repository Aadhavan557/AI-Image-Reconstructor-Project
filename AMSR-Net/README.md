# AMSR-Net: Adaptive Multi-Stage Semiconductor Restoration Network

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning pipeline and interactive web application designed for **semiconductor inspection image restoration**, combining **Denoising** and **2× Super-Resolution** to recover microscopic nanoscale circuit structures.

---

## 1. Project Title
**AI-Based Restoration of Degraded Images for Semiconductor Inspection using AMSR-Net**

---

## 2. Problem Statement
Semiconductor manufacturing relies on high-resolution Scanning Electron Microscope (SEM) images for quality control and defect detection. However, SEM images suffer from:
1. **High-frequency Speckle Noise** due to electron scattering.
2. **Spatial Resolution Reduction** to speed up scanning times.
3. **Loss of Fine Structural Details** (e.g., nanoscale wire edges and contacts).

Standard interpolation methods (such as Bicubic upsampling) fail to remove speckle noise and produce blurry edges, rendering automated defect detection algorithms unreliable.

---

## 3. Motivation
Restoring 128×128 degraded noisy images into 256×256 clear images allows semiconductor inspection tools to run at high speed (low resolution scanning) while maintaining micro-nanometer level precision through AI-assisted reconstruction.

---

## 4. Dataset
- **Ground Truth (GT):** High-resolution clean grayscale semiconductor SEM images (256×256).
- **NoisyLR:** Low-resolution (128×128) noisy grayscale images with severe speckle corruption.
- **Unseen Test Set:** 400 NoisyLR evaluation images in `dataset/NoisyLR`.

---

## 5. Data Format
The primary scientific data format is single-channel 2D **NumPy arrays (`.npy`)**, representing 32-bit floating-point grayscale electron intensity values. Standard image formats (`.png`, `.jpg`, `.tiff`) are also fully supported by the pipeline.

---

## 6. Preprocessing & Normalization
To prevent numerical instability caused by speckle noise outliers:
1. **Clipping:** Raw inputs are clipped to range `[-0.2, 1.2]`.
2. **Affine Scaling:** Scaled via `(x - 0.5) / 0.5` to range `[-1.4, +1.4]`.
3. **Ground Truth Normalization:** GT images are directly mapped to `[-1.0, +1.0]`.

---

## 7. AMSR-Net Architecture
AMSR-Net is a hybrid multi-stage deep learning architecture:

```text
Input NoisyLR (128x128)
       │
       ▼
Stage 0: Conv 3×3 Shallow Feature Extractor
       │
       ▼
Stage 1: CNN Residual Blocks (Local Feature Extraction)
       │
       ▼
Stage 2: Restormer MDTA Blocks (Global & Channel Attention)
       │
       ▼
Stage 3: Swin Transformer Pairs (Window-based Spatial Attention)
       │
       ▼
Stage 4: Feature Fusion Module (Combines local, global, and spatial features)
       │
       ▼
Stage 5: 2× PixelShuffle Upsampler (Resolution doubling to 256x256)
       │
       ▼
Stage 6: Global Bilinear Residual Skip Connection + Zero-Init Conv Head
       │
       ▼
Output Restored HR (256x256)
```

---

## 8–13. Key Components
- **CNN Component:** Extracts high-frequency local gradient details and spatial textures.
- **Restormer Component:** Multi-Dhead Transposed Attention (MDTA) blocks operate across channels to suppress global speckle noise.
- **Swin Transformer Component:** Shifted Window-based Multi-head Self-Attention (W-MSA) captures non-local spatial dependencies.
- **Feature Fusion:** Concatenates and projects feature maps from all three representations.
- **PixelShuffle:** Efficiently reorganizes channel depth into spatial resolution (2× upscaling) without checkerboard artifacts.
- **Global Residual Skip:** Adds a direct bilinear upsampled shortcut from input to output, allowing the network to only learn the high-frequency residual signal.

---

## 14. Loss Functions
The network is trained using a weighted composite loss function:
$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{Charbonnier}} + 0.2 \cdot \mathcal{L}_{\text{SSIM}} + 0.1 \cdot \mathcal{L}_{\text{Edge}}$$
- **Charbonnier Loss:** Robust $L_1$ variant ($\epsilon = 10^{-3}$) for smooth residual learning.
- **SSIM Loss:** Preserves structural coherence and visual quality.
- **Sobel Edge Loss:** Maximizes sharpness on circuit edges.

---

## 15. Training Setup
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $= 10^{-4}$)
- **Scheduler:** CosineAnnealingLR ($T_{\max}=100$, $\eta_{\min}=10^{-6}$)
- **Precision:** Mixed Precision (`torch.amp.autocast`) with float32 loss calculation.
- **Hardware:** NVIDIA RTX 2050 (4.3 GB VRAM).

---

## 16. Results & Benchmark Comparison

| Model | PSNR (dB) | SSIM | Params (M) | Inference Time (ms) |
|---|:---:|:---:|:---:|:---:|
| **Bicubic Baseline** | 25.79 | 0.647 | 0.00 | < 1.0 ms |
| **AMSR-Net (Ours)** | **29.25** | **0.795** | **~0.97** | **~49.0 ms** |

---

## 17. Inference
Run inference on single images or test sets using `phase10_inference.py` or the evaluation script.

---

## 18. Standalone Evaluation Script
Use `evaluation.py` for headless batch evaluation:

```bash
python evaluation.py --input path/to/NoisyLR --output path/to/output_dir
```

Optional flags:
- `--gt path/to/GT`: Computes PSNR/SSIM if GT images are available.
- `--device cuda|cpu`: Selects computing hardware.

---

## 19. Interactive Web Dashboard
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Features:
- Upload `.npy`, `.png`, `.jpg`, `.tiff` images.
- Side-by-side visual comparison and zoomed crop view.
- Real-time PSNR/SSIM metric calculation (when GT uploaded).
- Intensity distribution histograms, row/column pixel profiles, and Sobel edge maps.
- Direct download of restored images as PNG or NPY.

---

## 20. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/user/AMSR-Net.git
   cd AMSR-Net
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 21. Usage

- **Train Model:**
  ```bash
  python train.py --config config.py
  ```

- **Run Web App:**
  ```bash
  streamlit run app.py
  ```

- **Run Batch Evaluation:**
  ```bash
  python evaluation.py --input ../dataset/NoisyLR --output ./outputs/eval_results
  ```

---

## 22. Hardware Requirements
- **Minimum:** CPU with 8 GB RAM.
- **Recommended:** NVIDIA GPU with $\ge 4$ GB VRAM (e.g., RTX 2050 / GTX 1650).

---

## 23. Project Structure
```text
AMSR-Net/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusion rules
├── config.py                 # Hyperparameters & path configurations
├── train.py                  # Model training pipeline
├── evaluation.py             # Standalone CLI evaluation script
├── compare_models.py         # Dual-model benchmark comparison tool
├── app.py                    # Streamlit Web Application (Inference, Noise Lab, Dual Arena)
├── dataset/                  # Preprocessing & PyTorch dataloaders
├── models/                   # Network modules (AMSRNet, Restormer, Swin, Baseline)
├── losses/                   # Composite loss functions
├── metrics/                  # PSNR, SSIM, and Edge Preservation metrics
├── utils/                    # Trainer, Checkpoint, and Noise Generator utilities
├── scripts/                  # Phase verification and exploratory analysis scripts
├── weights/                  # Trained model checkpoints (best_checkpoint.pth)
├── logs/                     # Training logs & TensorBoard event logs
└── outputs/                  # Exported benchmark test suites & inference outputs
```

---

## 24. Limitations
- Fixed to 2× Super-Resolution factor.
- Requires input spatial dimensions to be multiples of 8 due to Swin W-MSA windowing.

---

## 25. Future Work
- Extend to 4× Super-Resolution with adaptive window padding.
- Quantize model to TensorRT/ONNX for sub-10ms edge inference on SEM hardware.
