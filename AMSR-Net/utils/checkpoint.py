"""
utils/checkpoint.py
====================
AMSR-Net - Phase 6: Checkpoint Save / Load
--------------------------------------------
Manages serialisation and deserialisation of all training state needed
to resume an interrupted training run exactly where it left off.

What is saved in a checkpoint?
--------------------------------
A complete checkpoint saves EVERYTHING needed to resume training:

  1. model_state_dict    : trained model weights
  2. optimizer_state_dict: optimiser momentum buffers and adaptive rates
                           (AdamW's m1, m2 moments; essential for continuity)
  3. scheduler_state_dict: LR scheduler state (current step, last LR)
  4. scaler_state_dict   : GradScaler state (loss scale value, growth interval)
                           Used only when mixed precision is enabled.
  5. epoch               : last completed epoch index
  6. best_psnr           : best validation PSNR so far
  7. best_ssim           : best validation SSIM so far
  8. config_snapshot     : snapshot of key hyperparameters at save time
                           (for diagnostic/reproducibility purposes)

Why save the optimiser state?
--------------------------------
Resuming with a fresh optimiser (e.g., from a saved model only) causes
a significant performance regression because:
  - AdamW's adaptive learning rates (m2 moment) need to warm up again.
  - Without momentum history (m1), the first few resumed epochs are
    effectively a cold restart, which wastes compute time.

Why save the GradScaler state?
--------------------------------
If we resume after a NaN loss event (which can happen with AMP at the
wrong loss scale), a fresh scaler will start with loss_scale=65536 and
may immediately underflow again. Restoring the prior stable scale value
avoids this training instability.

File naming convention
----------------------
  amsr_net_best.pth      : best validation PSNR checkpoint (for inference)
  amsr_net_epoch_{N}.pth : periodic checkpoint every CKPT_SAVE_EVERY epochs
  amsr_net_latest.pth    : always the most recent epoch (for resuming)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg


# ===========================================================================
# Save
# ===========================================================================

def save_checkpoint(
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler:  Any,
    scaler:     Optional[Any],
    epoch:      int,
    best_psnr:  float,
    best_ssim:  float,
    save_path:  str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a complete training checkpoint.

    Parameters
    ----------
    model      : nn.Module            Trained model.
    optimizer  : Optimizer            Current optimiser.
    scheduler  : LR scheduler         Current LR scheduler.
    scaler     : GradScaler or None   AMP grad scaler (None if FP32).
    epoch      : int                  Last completed epoch (0-indexed).
    best_psnr  : float                Best validation PSNR seen so far.
    best_ssim  : float                Best validation SSIM seen so far.
    save_path  : str                  Full path to .pth file.
    extra_meta : Dict or None         Any additional key-value pairs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    state = {
        "epoch":                epoch,
        "best_psnr":            best_psnr,
        "best_ssim":            best_ssim,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
        # Config snapshot for diagnostics
        "config_snapshot": {
            "lr":         cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "patch_size": cfg.PATCH_SIZE,
            "dim":        cfg.BASELINE_FEATURES,
            "num_blocks": cfg.BASELINE_NUM_BLOCKS,
            "scale":      cfg.SCALE,
        },
    }

    if extra_meta:
        state["extra_meta"] = extra_meta

    torch.save(state, save_path)


# ===========================================================================
# Load
# ===========================================================================

def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any]                   = None,
    scaler:    Optional[Any]                   = None,
    device:    Optional[torch.device]          = None,
    strict:    bool                            = True,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Parameters
    ----------
    path      : str            Path to the .pth file.
    model     : nn.Module      Model to load weights into.
    optimizer : Optimizer      Restore optimiser state if provided.
    scheduler : LR scheduler   Restore scheduler state if provided.
    scaler    : GradScaler     Restore scaler state if provided.
    device    : torch.device   Map location for tensors. Auto-detected if None.
    strict    : bool           Whether to strictly match model state dict keys.

    Returns
    -------
    Dict with keys: "epoch", "best_psnr", "best_ssim", "config_snapshot".

    Raises
    ------
    FileNotFoundError  If the checkpoint file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_loc = device if device else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    state = torch.load(path, map_location=map_loc, weights_only=False)

    # Load model weights
    model.load_state_dict(state["model_state_dict"], strict=strict)

    # Restore optimiser state (preserves momentum, adaptive rates)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    # Restore scheduler state (preserves LR schedule position)
    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    # Restore AMP scaler state
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])

    return {
        "epoch":           state.get("epoch", 0),
        "best_psnr":       state.get("best_psnr", -1.0),
        "best_ssim":       state.get("best_ssim", -1.0),
        "config_snapshot": state.get("config_snapshot", {}),
    }


# ===========================================================================
# Naming helpers
# ===========================================================================

def best_ckpt_path() -> str:
    """Return the path for the best PSNR checkpoint."""
    return os.path.join(cfg.WEIGHTS_DIR, "amsr_net_best.pth")


def latest_ckpt_path() -> str:
    """Return the path for the latest (most recent epoch) checkpoint."""
    return os.path.join(cfg.WEIGHTS_DIR, "amsr_net_latest.pth")


def epoch_ckpt_path(epoch: int) -> str:
    """Return the path for a periodic epoch checkpoint."""
    return os.path.join(cfg.WEIGHTS_DIR, f"amsr_net_epoch_{epoch:04d}.pth")
