"""
models/config.py
================
Submission-adapted configuration for AMSR-Net.

All dataset paths are removed — they are not needed for inference.
All model hyper-parameters, normalisation constants, and the weights
path are preserved exactly as used during training.
"""

import os
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this file = models/)
# ---------------------------------------------------------------------------
BASE_DIR:    str = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR: str = BASE_DIR   # weights (.pth) live alongside this config
OUTPUTS_DIR: str = os.path.join(os.path.dirname(BASE_DIR), "outputs")

# ---------------------------------------------------------------------------
# Image / Dataset properties
# ---------------------------------------------------------------------------
LR_SIZE:  Tuple[int, int] = (128, 128)   # NoisyLR spatial dimensions
GT_SIZE:  Tuple[int, int] = (256, 256)   # Ground-truth spatial dimensions
SCALE:    int             = 2            # 2× super-resolution
CHANNELS: int             = 1            # Grayscale
DTYPE:    str             = "float32"

# ---------------------------------------------------------------------------
# Data splitting & loading (used by dataset module; not needed for inference)
# ---------------------------------------------------------------------------
VAL_SPLIT:   float = 0.10
RANDOM_SEED: int   = 42

PATCH_SIZE:  int  = 64
BATCH_SIZE:  int  = 16
NUM_WORKERS: int  = 0
PIN_MEMORY:  bool = True

# ---------------------------------------------------------------------------
# Augmentation probabilities (training only)
# ---------------------------------------------------------------------------
AUG_HFLIP_P: float = 0.5
AUG_VFLIP_P: float = 0.5
AUG_ROT90_P: float = 0.5

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
NORM_MEAN: float = 0.5
NORM_STD:  float = 0.5

# ---------------------------------------------------------------------------
# Baseline CNN
# ---------------------------------------------------------------------------
BASELINE_FEATURES:   int = 64
BASELINE_NUM_BLOCKS: int = 8

# ---------------------------------------------------------------------------
# Restormer
# ---------------------------------------------------------------------------
RESTORMER_DIM:            int   = 48
RESTORMER_NUM_BLOCKS:     List  = [4, 6, 6, 8]
RESTORMER_NUM_REFINEMENT: int   = 4
RESTORMER_NUM_HEADS:      List  = [1, 2, 4, 8]
RESTORMER_FFN_EXPANSION:  float = 2.66
RESTORMER_BIAS:           bool  = False

# ---------------------------------------------------------------------------
# SwinIR
# ---------------------------------------------------------------------------
SWINIR_UPSCALE:         int   = SCALE
SWINIR_IN_CH:           int   = CHANNELS
SWINIR_IMG_SIZE:        int   = LR_SIZE[0]
SWINIR_WINDOW_SIZE:     int   = 8
SWINIR_IMG_RANGE:       float = 1.0
SWINIR_EMBED_DIM:       int   = 60
SWINIR_DEPTHS:          List  = [6, 6, 6, 6]
SWINIR_NUM_HEADS:       List  = [6, 6, 6, 6]
SWINIR_MLP_RATIO:       float = 2.0
SWINIR_UPSAMPLER:       str   = "pixelshuffle"
SWINIR_RESI_CONNECTION: str   = "1conv"

# ---------------------------------------------------------------------------
# AMSR-Net hybrid (core inference configuration)
# ---------------------------------------------------------------------------
AMSRNET_DIM:              int = 64
AMSRNET_ENCODER_BLOCKS:   int = 4
AMSRNET_DECODER_BLOCKS:   int = 4
AMSRNET_RESTORMER_BLOCKS: int = 6
AMSRNET_SWIN_BLOCKS:      int = 4
AMSRNET_WINDOW_SIZE:      int = 8
AMSRNET_NUM_HEADS:        int = 4

# ---------------------------------------------------------------------------
# Loss weights (training only; kept for reference)
# ---------------------------------------------------------------------------
LOSS_L1_W:          float = 1.0
LOSS_SSIM_W:        float = 0.2
LOSS_EDGE_W:        float = 0.1
LOSS_CHARBONNIER_W: float = 0.5

# ---------------------------------------------------------------------------
# Training settings (kept for reference; not used at inference)
# ---------------------------------------------------------------------------
NUM_EPOCHS:          int   = 100
LEARNING_RATE:       float = 2e-4
WEIGHT_DECAY:        float = 1e-4
BETAS:               Tuple = (0.9, 0.999)
GRAD_CLIP_NORM:      float = 1.0
MIXED_PRECISION:     bool  = True
SCHEDULER_T_MAX:     int   = NUM_EPOCHS
SCHEDULER_ETA_MIN:   float = 1e-7
EARLY_STOP_PATIENCE: int   = 15
CKPT_SAVE_EVERY:     int   = 10

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
BEST_MODEL_PATH: str = os.path.join(WEIGHTS_DIR, "amsr_net_best.pth")
