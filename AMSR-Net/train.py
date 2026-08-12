"""
train.py
=========
AMSR-Net - Main Training Entry Point
--------------------------------------
Ties together all phases to run full model training.

Usage
-----
  # Train from scratch:
  python train.py

  # Resume from latest checkpoint:
  python train.py --resume

  # Resume from specific checkpoint:
  python train.py --resume --ckpt weights/amsr_net_epoch_0010.pth

  # Override epochs for quick test:
  python train.py --epochs 5

  # Use different model (future phases):
  python train.py --model baseline

Monitor training with TensorBoard:
  tensorboard --logdir AMSR-Net/logs
"""

import sys
import os
import argparse
from pathlib import Path

# UTF-8 Windows fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import config as cfg
from models          import BaselineCNN, AMSRNet
from dataset         import build_dataloaders
from utils.trainer   import Trainer
from utils.checkpoint import latest_ckpt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMSR-Net Training Script"
    )
    parser.add_argument(
        "--model", type=str, default="amsrnet",
        choices=["baseline", "amsrnet"],
        help="Model architecture to train: 'baseline' (Phase 3 CNN) or 'amsrnet' (Phase 7 Hybrid) (default: amsrnet)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override cfg.NUM_EPOCHS (useful for quick tests)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to a specific checkpoint to resume from"
    )
    parser.add_argument(
        "--workers", type=int, default=cfg.NUM_WORKERS,
        help=f"DataLoader worker processes (default: {cfg.NUM_WORKERS})"
    )
    parser.add_argument(
        "--batch", type=int, default=cfg.BATCH_SIZE,
        help=f"Batch size (default: {cfg.BATCH_SIZE})"
    )
    parser.add_argument(
        "--run-name", type=str, default="amsrnet_run1",
        help="TensorBoard run name (default: amsrnet_run1)"
    )
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    """Instantiate the requested model."""
    if model_name == "baseline":
        return BaselineCNN(
            in_channels = cfg.CHANNELS,
            dim         = cfg.BASELINE_FEATURES,
            num_blocks  = cfg.BASELINE_NUM_BLOCKS,
            scale       = cfg.SCALE,
        )
    elif model_name == "amsrnet":
        return AMSRNet(
            in_channels      = cfg.CHANNELS,
            dim              = cfg.AMSRNET_DIM,
            encoder_blocks   = cfg.AMSRNET_ENCODER_BLOCKS,
            restormer_blocks = cfg.AMSRNET_RESTORMER_BLOCKS,
            swin_blocks      = cfg.AMSRNET_SWIN_BLOCKS,
            num_heads        = cfg.AMSRNET_NUM_HEADS,
            window_size      = cfg.AMSRNET_WINDOW_SIZE,
            scale            = cfg.SCALE,
        )
    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    print("\n  Building data loaders...")
    train_loader, val_loader = build_dataloaders(
        batch_size  = args.batch,
        num_workers = args.workers,
    )
    print(f"  Train: {len(train_loader.dataset):,} images  "
          f"({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} images  "
          f"({len(val_loader)} batches)")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\n  Building model: {args.model}...")
    model   = build_model(args.model)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_param:,}  ({n_param/1e6:.2f}M)")

    # ------------------------------------------------------------------
    # Resume checkpoint path
    # ------------------------------------------------------------------
    resume_path = None
    if args.ckpt:
        resume_path = args.ckpt
    elif args.resume:
        candidate = latest_ckpt_path()
        if os.path.exists(candidate):
            resume_path = candidate
            print(f"\n  Resuming from: {resume_path}")
        else:
            print(f"\n  [WARN] --resume specified but no checkpoint found at {candidate}. "
                  f"Training from scratch.")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        run_name     = args.run_name,
        resume_from  = resume_path,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(num_epochs=args.epochs)


if __name__ == "__main__":
    # Windows multiprocessing safety guard for DataLoader workers
    import multiprocessing
    multiprocessing.freeze_support()
    main()
