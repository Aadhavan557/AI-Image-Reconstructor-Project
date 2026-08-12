"""
config.py
=========
Central configuration file for the AMSR-Net project.

Every hyperparameter, path, and training setting lives here so that
the rest of the codebase never contains magic numbers.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Dataset paths — adjust if your layout differs
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Root of the raw dataset (relative to project root)
DATASET_ROOT: str = os.path.join(
    os.path.dirname(BASE_DIR), "dataset"  # d:/aadhavan/AI Image Reconstructor Project/dataset
)

TRAIN_GT_DIR:      str = os.path.join(DATASET_ROOT, "train", "GT")
TRAIN_NOISYLR_DIR: str = os.path.join(DATASET_ROOT, "train", "NoisyLR")
TEST_NOISYLR_DIR:  str = os.path.join(DATASET_ROOT, "NoisyLR")

# Output directories (created automatically)
WEIGHTS_DIR: str = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR: str = os.path.join(BASE_DIR, "outputs")
LOGS_DIR:    str = os.path.join(BASE_DIR, "logs")

for _d in [WEIGHTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Image / Dataset properties (confirmed from Phase 1 exploration)
# ---------------------------------------------------------------------------
LR_SIZE:    Tuple[int, int] = (128, 128)   # NoisyLR spatial dimensions
GT_SIZE:    Tuple[int, int] = (256, 256)   # Ground-truth spatial dimensions
SCALE:      int             = 2            # Super-resolution upscaling factor
CHANNELS:   int             = 1            # Grayscale → single channel
DTYPE:      str             = "float32"


# ---------------------------------------------------------------------------
# Data splitting & loading
# ---------------------------------------------------------------------------
VAL_SPLIT:   float = 0.10   # 10 % of training data used for validation
RANDOM_SEED: int   = 42     # Reproducibility seed

PATCH_SIZE:  int = 64       # Random crop size from the LR image (GT patch = 128)
BATCH_SIZE:  int = 16       # Images per mini-batch
NUM_WORKERS: int = 0        # 0 prevents WinError 1455 paging file crash on Windows
PIN_MEMORY:  bool = True    # Pin CPU memory for faster GPU transfer


# ---------------------------------------------------------------------------
# Augmentation probabilities
# ---------------------------------------------------------------------------
AUG_HFLIP_P:  float = 0.5    # Horizontal flip probability
AUG_VFLIP_P:  float = 0.5    # Vertical flip probability
AUG_ROT90_P:  float = 0.5    # 90° rotation probability


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
# GT images are already in [0, 1].
# NoisyLR images can fall slightly outside [0, 1] due to noise.
# We clip and normalise to [-1, 1] before feeding to the network.
NORM_MEAN: float = 0.5
NORM_STD:  float = 0.5


# ---------------------------------------------------------------------------
# Baseline CNN (Phase 3)
# ---------------------------------------------------------------------------
BASELINE_FEATURES:   int = 64
BASELINE_NUM_BLOCKS: int = 8


# ---------------------------------------------------------------------------
# Restormer (Phase 4)
# ---------------------------------------------------------------------------
RESTORMER_DIM:           int       = 48
RESTORMER_NUM_BLOCKS:    List[int] = field(default_factory=lambda: [4, 6, 6, 8])
RESTORMER_NUM_REFINEMENT:int       = 4
RESTORMER_NUM_HEADS:     List[int] = field(default_factory=lambda: [1, 2, 4, 8])
RESTORMER_FFN_EXPANSION:  float    = 2.66
RESTORMER_BIAS:          bool      = False


# ---------------------------------------------------------------------------
# SwinIR (Phase 5)
# ---------------------------------------------------------------------------
SWINIR_UPSCALE:       int = SCALE
SWINIR_IN_CH:         int = CHANNELS
SWINIR_IMG_SIZE:      int = LR_SIZE[0]
SWINIR_WINDOW_SIZE:   int = 8
SWINIR_IMG_RANGE:     float = 1.0
SWINIR_EMBED_DIM:     int = 60
SWINIR_DEPTHS:        List[int] = field(default_factory=lambda: [6, 6, 6, 6])
SWINIR_NUM_HEADS:     List[int] = field(default_factory=lambda: [6, 6, 6, 6])
SWINIR_MLP_RATIO:     float = 2.0
SWINIR_UPSAMPLER:     str = "pixelshuffle"
SWINIR_RESI_CONNECTION: str = "1conv"


# ---------------------------------------------------------------------------
# AMSR-Net hybrid (Phases 6 & 7)
# ---------------------------------------------------------------------------
AMSRNET_DIM:              int = 64
AMSRNET_ENCODER_BLOCKS:   int = 4
AMSRNET_DECODER_BLOCKS:   int = 4
AMSRNET_RESTORMER_BLOCKS: int = 6
AMSRNET_SWIN_BLOCKS:      int = 4
AMSRNET_WINDOW_SIZE:      int = 8
AMSRNET_NUM_HEADS:        int = 4


# ---------------------------------------------------------------------------
# Loss weights (Phase 8)
# ---------------------------------------------------------------------------
LOSS_L1_W:          float = 1.0
LOSS_SSIM_W:        float = 0.2
LOSS_EDGE_W:        float = 0.1
LOSS_CHARBONNIER_W: float = 0.5


# ---------------------------------------------------------------------------
# Training (Phase 9)
# ---------------------------------------------------------------------------
NUM_EPOCHS:         int   = 100
LEARNING_RATE:      float = 2e-4
WEIGHT_DECAY:       float = 1e-4
BETAS:              Tuple[float, float] = (0.9, 0.999)

GRAD_CLIP_NORM:     float = 1.0   # Max gradient L2 norm for clipping
MIXED_PRECISION:    bool  = True  # Use torch.cuda.amp (FP16/BF16)

SCHEDULER_T_MAX:    int   = NUM_EPOCHS  # CosineAnnealingLR cycle length
SCHEDULER_ETA_MIN:  float = 1e-7        # Minimum LR at end of cosine cycle

# Early stopping
EARLY_STOP_PATIENCE: int = 15   # Stop if val PSNR doesn't improve for N epochs

# Checkpoint interval (save every N epochs in addition to best)
CKPT_SAVE_EVERY: int = 10


# ---------------------------------------------------------------------------
# Inference (Phase 11)
# ---------------------------------------------------------------------------
BEST_MODEL_PATH: str = os.path.join(WEIGHTS_DIR, "amsr_net_best.pth")
