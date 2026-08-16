"""
dataset/dataloader.py
======================
AMSR-Net - Phase 2: DataLoader Factory
----------------------------------------
Provides a single entry-point function `build_dataloaders()` that:

  1. Scans the dataset directories.
  2. Applies a reproducible train/validation split.
  3. Constructs SemiconductorDataset instances for train and val.
  4. Returns PyTorch DataLoaders ready to be consumed by the trainer.

Design decisions
----------------
- Train/val split is file-level (not patch-level).
  Patch-level splits would leak information across the boundary because
  patches from the same image share the same noise characteristics.

- Shuffle is applied to the training file list BEFORE splitting so that
  both splits are representative of the full dataset distribution.

- A fixed random seed ensures reproducibility: the same split is used
  across all training runs, making ablation studies comparable.

- num_workers is set via cfg.NUM_WORKERS. On Windows, multiprocessing
  with DataLoader requires the __main__ guard (spawn start method).
  The factory function itself is safe to call from any context.
"""

import os
import sys
import random
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project root import
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from dataset.semiconductor_dataset import SemiconductorDataset, scan_directory


# ===========================================================================
# Public API
# ===========================================================================

def build_dataloaders(
    train_gt_dir:      str   = cfg.TRAIN_GT_DIR,
    train_noisylr_dir: str   = cfg.TRAIN_NOISYLR_DIR,
    val_split:         float = cfg.VAL_SPLIT,
    patch_size:        int   = cfg.PATCH_SIZE,
    scale:             int   = cfg.SCALE,
    batch_size:        int   = cfg.BATCH_SIZE,
    num_workers:       int   = cfg.NUM_WORKERS,
    pin_memory:        bool  = cfg.PIN_MEMORY,
    seed:              int   = cfg.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader) for the semiconductor dataset.

    Parameters
    ----------
    train_gt_dir      : str    Path to train/GT/ directory.
    train_noisylr_dir : str    Path to train/NoisyLR/ directory.
    val_split         : float  Fraction of files reserved for validation.
    patch_size        : int    LR patch side length (GT = patch_size * scale).
    scale             : int    Super-resolution scale factor.
    batch_size        : int    Samples per mini-batch.
    num_workers       : int    DataLoader worker processes.
    pin_memory        : bool   Pin CPU memory for faster GPU transfer.
    seed              : int    RNG seed for reproducible split.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)

    Notes on split strategy
    -----------------------
    With 3,200 images and val_split=0.10:
        Train: 2,880 images  (each yields many random crops per epoch)
        Val:   320  images   (centre-cropped, deterministic)

    With patch_size=64 and LR images of 128x128:
        Max non-overlapping patches per image: (128/64)^2 = 4
        Effective train size per epoch: ~11,520 patches (before augmentation)
        Augmentation multiplier: up to 8x (hflip x vflip x {0,90,180,270})
    """
    # -----------------------------------------------------------------------
    # 1. Scan and pair files
    # -----------------------------------------------------------------------
    gt_files = scan_directory(train_gt_dir)
    lr_files = scan_directory(train_noisylr_dir)

    assert len(gt_files) == len(lr_files) > 0, (
        f"Expected equal non-zero file counts. "
        f"GT: {len(gt_files)}, LR: {len(lr_files)}"
    )

    # -----------------------------------------------------------------------
    # 2. Reproducible shuffle + split
    # -----------------------------------------------------------------------
    rng = random.Random(seed)

    # Pair, shuffle, then unpack -- preserves GT/LR correspondence
    pairs = list(zip(gt_files, lr_files))
    rng.shuffle(pairs)

    n_val   = max(1, int(len(pairs) * val_split))
    n_train = len(pairs) - n_val

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    train_gt, train_lr = zip(*train_pairs)
    val_gt,   val_lr   = zip(*val_pairs)

    # -----------------------------------------------------------------------
    # 3. Build Dataset instances
    # -----------------------------------------------------------------------
    train_dataset = SemiconductorDataset(
        gt_files   = list(train_gt),
        lr_files   = list(train_lr),
        patch_size = patch_size,
        scale      = scale,
        is_train   = True,
    )

    val_dataset = SemiconductorDataset(
        gt_files   = list(val_gt),
        lr_files   = list(val_lr),
        patch_size = patch_size,
        scale      = scale,
        is_train   = False,
    )

    # -----------------------------------------------------------------------
    # 4. Build DataLoaders
    # -----------------------------------------------------------------------
    train_loader = DataLoader(
        dataset     = train_dataset,
        batch_size  = batch_size,
        shuffle     = True,      # Re-shuffle file order each epoch
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,      # Drop incomplete final batch for stable BN stats
        persistent_workers = (num_workers > 0),
    )

    val_loader = DataLoader(
        dataset     = val_dataset,
        batch_size  = batch_size,
        shuffle     = False,     # Deterministic validation order
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
        persistent_workers = (num_workers > 0),
    )

    return train_loader, val_loader


def build_test_dataloader(
    test_noisylr_dir: str  = cfg.TEST_NOISYLR_DIR,
    patch_size:       int  = cfg.PATCH_SIZE,
    scale:            int  = cfg.SCALE,
    batch_size:       int  = 1,
    num_workers:      int  = cfg.NUM_WORKERS,
    pin_memory:       bool = cfg.PIN_MEMORY,
) -> DataLoader:
    """
    Build a DataLoader for the test set (NoisyLR only -- no GT).

    At test time there are no GT files. We create a special dataset
    using the same LR file as both GT and LR placeholder, then discard
    the "gt" key at inference time.

    Parameters
    ----------
    test_noisylr_dir : str   Path to test/NoisyLR/ directory.
    patch_size       : int   LR patch size (used for centre crop).
    scale            : int   Upscaling factor.
    batch_size       : int   Usually 1 for inference (tile-based).
    num_workers      : int   DataLoader workers.
    pin_memory       : bool  Pin CPU memory.

    Returns
    -------
    DataLoader
        Yields batches with keys "lr" and "filename".
        (The "gt" key will be a dummy placeholder -- ignore it at inference.)
    """
    lr_files = scan_directory(test_noisylr_dir)

    assert len(lr_files) > 0, f"No .npy files found in {test_noisylr_dir}"

    # For test set, pass LR files as both GT and LR (GT is unused at inference)
    test_dataset = SemiconductorDataset(
        gt_files   = lr_files,   # placeholder -- not used at inference
        lr_files   = lr_files,
        patch_size = patch_size,
        scale      = scale,
        is_train   = False,      # centre crop, no augmentation
    )

    test_loader = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    return test_loader
