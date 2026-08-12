"""
utils/trainer.py
=================
AMSR-Net - Phase 6: Trainer
-----------------------------
The Trainer class encapsulates the complete training loop for AMSR-Net,
including training, validation, checkpointing, and logging.

Design Philosophy
-----------------
The Trainer is model-agnostic: it accepts any nn.Module that maps
(B, 1, H, W) -> (B, 1, 2H, 2W). This means the same trainer can be used
for the BaselineCNN (Phase 3) and the full AMSR-Net hybrid (Phases 6-7).

Key features:
  1. Mixed Precision (AMP): uses torch.cuda.amp for FP16 forward/backward.
     Reduces GPU memory by ~50% and speeds up training on RTX 2050 (Ampere).
  2. Gradient Clipping: clips gradients at cfg.GRAD_CLIP_NORM (L2 norm = 1.0)
     to prevent explosion, which is common early in training with deep residual
     networks and Charbonnier + SSIM composite losses.
  3. Cosine Annealing LR: decays LR from cfg.LEARNING_RATE to cfg.SCHEDULER_ETA_MIN
     over cfg.NUM_EPOCHS. This improves final convergence vs. step decay.
  4. TensorBoard: logs train/val loss, PSNR, SSIM, LR, and gradient norm
     every epoch. Open with: tensorboard --logdir AMSR-Net/logs
  5. Early Stopping: stops training if val PSNR has not improved for
     cfg.EARLY_STOP_PATIENCE epochs.
  6. Checkpointing: saves "best" and "latest" checkpoints automatically,
     plus periodic epoch checkpoints every cfg.CKPT_SAVE_EVERY epochs.

Training Loop Per Epoch
-----------------------
  1. model.train()
  2. for each batch:
       a. Move lr, gt to device
       b. AMP autocast forward pass
       c. CompositeLoss (Charbonnier + SSIM + Edge)
       d. scaler.scale(loss).backward()
       e. scaler.unscale_(optimizer)  [for gradient clipping]
       f. nn.utils.clip_grad_norm_(parameters, max_norm)
       g. scaler.step(optimizer)
       h. scaler.update()
       i. scheduler.step() [per-epoch]
  3. Validation:
       a. model.eval(), torch.no_grad()
       b. Denormalise predictions to [0, 1]
       c. Accumulate PSNR and SSIM via MetricTracker
  4. Log to TensorBoard
  5. Checkpoint if improved

Why AMP (Automatic Mixed Precision)?
--------------------------------------
  - Forward pass uses FP16 (half precision): 2x faster matrix multiply on RTX
  - Loss and gradients are accumulated in FP32 to maintain numerical stability
  - GradScaler dynamically adjusts the loss scale to prevent FP16 underflow
  - Net effect: ~1.5-2x faster training, ~40% less VRAM on RTX 2050
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from losses    import CompositeLoss
from metrics   import MetricTracker
from dataset.semiconductor_dataset import denormalise
from utils.checkpoint import (
    save_checkpoint,
    best_ckpt_path,
    latest_ckpt_path,
    epoch_ckpt_path,
)


class Trainer:
    """
    Model-agnostic training engine for AMSR-Net.

    Parameters
    ----------
    model        : nn.Module       The model to train.
    train_loader : DataLoader      Training data loader.
    val_loader   : DataLoader      Validation data loader.
    device       : torch.device    Training device (CPU or CUDA).
    run_name     : str             TensorBoard run identifier.
    resume_from  : str or None     Path to checkpoint to resume from.
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        run_name:     str = "baseline",
        resume_from:  Optional[str] = None,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.run_name     = run_name

        # ------------------------------------------------------------------
        # Loss function
        # ------------------------------------------------------------------
        self.criterion = CompositeLoss(
            w_charb    = cfg.LOSS_CHARBONNIER_W,
            w_ssim     = cfg.LOSS_SSIM_W,
            w_edge     = cfg.LOSS_EDGE_W,
            channels   = cfg.CHANNELS,
            data_range = 2.0,    # images normalised to [-1, +1]
        ).to(device)

        # ------------------------------------------------------------------
        # Optimiser: AdamW
        # AdamW decouples weight decay from gradient update (Loshchilov 2019).
        # Compared to Adam + L2 regularisation, AdamW applies weight decay
        # uniformly regardless of the gradient scale -- important for
        # parameters with different learning rate scales (attention vs. conv).
        # ------------------------------------------------------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = cfg.LEARNING_RATE,
            betas        = cfg.BETAS,
            weight_decay = cfg.WEIGHT_DECAY,
        )

        # ------------------------------------------------------------------
        # LR Scheduler: CosineAnnealingLR
        # Decays LR according to cosine curve: LR(t) = eta_min +
        #   0.5*(eta_max - eta_min) * (1 + cos(pi*t/T_max))
        # - Avoids abrupt LR drops that destabilise training
        # - Low LR at end of training allows fine-grained convergence
        # ------------------------------------------------------------------
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max   = cfg.SCHEDULER_T_MAX,
            eta_min = cfg.SCHEDULER_ETA_MIN,
        )

        # ------------------------------------------------------------------
        # AMP GradScaler (only active if CUDA is available)
        # ------------------------------------------------------------------
        self.use_amp = cfg.MIXED_PRECISION and device.type == "cuda"
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ------------------------------------------------------------------
        # Metric tracker
        # ------------------------------------------------------------------
        self.metric_tracker = MetricTracker(device=device)

        # ------------------------------------------------------------------
        # TensorBoard
        # ------------------------------------------------------------------
        log_dir = os.path.join(cfg.LOGS_DIR, run_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        # ------------------------------------------------------------------
        # Training state
        # ------------------------------------------------------------------
        self.start_epoch = 0
        self.best_psnr   = -1.0
        self.best_ssim   = -1.0

        # ------------------------------------------------------------------
        # Optionally resume from checkpoint
        # ------------------------------------------------------------------
        if resume_from is not None:
            self._resume(resume_from)

    # ======================================================================
    # Private: resume from checkpoint
    # ======================================================================

    def _resume(self, path: str) -> None:
        """Load checkpoint and restore all training state."""
        from utils.checkpoint import load_checkpoint
        meta = load_checkpoint(
            path      = path,
            model     = self.model,
            optimizer = self.optimizer,
            scheduler = self.scheduler,
            scaler    = self.scaler if self.use_amp else None,
            device    = self.device,
        )
        self.start_epoch = meta["epoch"] + 1
        self.best_psnr   = meta["best_psnr"]
        self.best_ssim   = meta["best_ssim"]
        self.metric_tracker.best_psnr  = self.best_psnr
        self.metric_tracker.best_ssim  = self.best_ssim
        self.metric_tracker.best_epoch = meta["epoch"]
        print(f"  [RESUME] Restored from epoch {meta['epoch']}  "
              f"(best PSNR={self.best_psnr:.2f} dB)")

    # ======================================================================
    # Private: single training epoch
    # ======================================================================

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one full training epoch.

        Returns
        -------
        Dict with keys: "loss", "charb", "ssim", "edge",
                        "grad_norm", "lr".
        """
        self.model.train()
        total_loss  = 0.0
        total_charb = 0.0
        total_ssim  = 0.0
        total_edge  = 0.0
        total_gnorm = 0.0
        n_batches   = 0

        iterable = (
            tqdm(self.train_loader,
                 desc=f"Epoch {epoch:04d} [TRAIN]",
                 leave=False, ncols=80)
            if HAS_TQDM else self.train_loader
        )

        for batch in iterable:
            lr_img = batch["lr"].to(self.device, non_blocking=True)
            gt_img = batch["gt"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass (AMP autocast)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred   = self.model(lr_img)
            
            # Compute losses safely outside of autocast in FP32
            # This completely prevents any FP16 overflow issues (inf/nan) in SSIM or Edge Loss
            losses = self.criterion.forward_detailed(pred.float(), gt_img.float())
            loss   = losses["total"]

            # Backward pass (scaled for AMP)
            self.scaler.scale(loss).backward()

            # Gradient clipping (BEFORE scaler.step)
            # Must unscale first so clip_grad_norm_ sees true gradient values
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=cfg.GRAD_CLIP_NORM
            ).item()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate for epoch-level average
            total_loss  += losses["total"].item()
            total_charb += losses["charb"].item()
            total_ssim  += losses["ssim"].item()
            total_edge  += losses["edge"].item()
            total_gnorm += grad_norm
            n_batches   += 1

        n = max(n_batches, 1)
        return {
            "loss":      total_loss  / n,
            "charb":     total_charb / n,
            "ssim":      total_ssim  / n,
            "edge":      total_edge  / n,
            "grad_norm": total_gnorm / n,
            "lr":        self.scheduler.get_last_lr()[0],
        }

    # ======================================================================
    # Private: validation epoch
    # ======================================================================

    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Run one full validation epoch.

        Returns
        -------
        Dict with keys: "psnr", "ssim", "loss".
        """
        self.model.eval()
        self.metric_tracker.reset()

        total_loss = 0.0
        n_batches  = 0

        iterable = (
            tqdm(self.val_loader,
                 desc=f"Epoch {epoch:04d} [VAL  ]",
                 leave=False, ncols=80)
            if HAS_TQDM else self.val_loader
        )

        with torch.no_grad():
            for batch in iterable:
                lr_img = batch["lr"].to(self.device, non_blocking=True)
                gt_img = batch["gt"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred     = self.model(lr_img)
                    
                val_loss = self.criterion(pred.float(), gt_img.float())

                total_loss += val_loss.item()
                n_batches  += 1

                # Denormalise from [-1,+1] to [0,1] before computing PSNR/SSIM.
                # Cast to float32: AMP leaves pred as FP16, which causes
                # dtype mismatch with the metric's Gaussian buffer (always FP32).
                pred_01 = denormalise(pred.float()).clamp(0.0, 1.0)
                gt_01   = denormalise(gt_img.float()).clamp(0.0, 1.0)
                self.metric_tracker.update(pred_01, gt_01)

        stats = self.metric_tracker.compute()
        stats["loss"] = total_loss / max(n_batches, 1)
        return stats

    # ======================================================================
    # Private: TensorBoard logging
    # ======================================================================

    def _log(
        self,
        epoch:       int,
        train_stats: Dict[str, float],
        val_stats:   Dict[str, float],
    ) -> None:
        """Write all metrics to TensorBoard."""
        w = self.writer

        # Training losses
        w.add_scalar("Loss/Train/Total",     train_stats["loss"],      epoch)
        w.add_scalar("Loss/Train/Charb",     train_stats["charb"],     epoch)
        w.add_scalar("Loss/Train/SSIM",      train_stats["ssim"],      epoch)
        w.add_scalar("Loss/Train/Edge",      train_stats["edge"],      epoch)
        w.add_scalar("Train/GradNorm",       train_stats["grad_norm"], epoch)
        w.add_scalar("Train/LR",             train_stats["lr"],        epoch)

        # Validation
        w.add_scalar("Loss/Val/Total",       val_stats["loss"],        epoch)
        w.add_scalar("Metrics/Val/PSNR_dB",  val_stats["psnr"],        epoch)
        w.add_scalar("Metrics/Val/SSIM",     val_stats["ssim"],        epoch)

        # Best so far
        w.add_scalar("Metrics/Best/PSNR_dB", self.best_psnr,           epoch)
        w.add_scalar("Metrics/Best/SSIM",    self.best_ssim,           epoch)

    # ======================================================================
    # Private: checkpoint management
    # ======================================================================

    def _save(self, epoch: int, is_best: bool) -> None:
        """Save latest checkpoint, best checkpoint, and periodic checkpoints."""

        def _do_save(path: str) -> None:
            save_checkpoint(
                model     = self.model,
                optimizer = self.optimizer,
                scheduler = self.scheduler,
                scaler    = self.scaler if self.use_amp else None,
                epoch     = epoch,
                best_psnr = self.best_psnr,
                best_ssim = self.best_ssim,
                save_path = path,
            )

        # Always save latest
        _do_save(latest_ckpt_path())

        # Save best
        if is_best:
            _do_save(best_ckpt_path())

        # Periodic epoch checkpoint
        if (epoch + 1) % cfg.CKPT_SAVE_EVERY == 0:
            _do_save(epoch_ckpt_path(epoch))

    # ======================================================================
    # Public: main training loop
    # ======================================================================

    def fit(self, num_epochs: Optional[int] = None) -> None:
        """
        Run the full training loop.

        Parameters
        ----------
        num_epochs : int or None  Override cfg.NUM_EPOCHS if provided.
        """
        total_epochs  = num_epochs if num_epochs is not None else cfg.NUM_EPOCHS
        patience      = cfg.EARLY_STOP_PATIENCE
        n_params      = sum(p.numel() for p in self.model.parameters()
                            if p.requires_grad)

        print("=" * 70)
        print(f"  Training {self.model.__class__.__name__}"
              f"  ({n_params/1e6:.2f}M params)")
        print(f"  Device       : {self.device}")
        print(f"  AMP          : {self.use_amp}")
        print(f"  Epochs       : {self.start_epoch} -> {total_epochs - 1}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val   batches: {len(self.val_loader)}")
        print(f"  TensorBoard  : tensorboard --logdir {cfg.LOGS_DIR}")
        print("=" * 70)

        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.perf_counter()

            # ---- Train ----
            train_stats = self._train_epoch(epoch)
            self.scheduler.step()

            # ---- Validate ----
            val_stats = self._val_epoch(epoch)

            epoch_time = time.perf_counter() - epoch_start

            # ---- Update best ----
            improved = self.metric_tracker.update_best(epoch)
            is_best  = improved["psnr_improved"]

            if val_stats["psnr"] > self.best_psnr:
                self.best_psnr = val_stats["psnr"]
            if val_stats["ssim"] > self.best_ssim:
                self.best_ssim = val_stats["ssim"]

            # ---- Log ----
            self._log(epoch, train_stats, val_stats)

            # ---- Checkpoint ----
            self._save(epoch, is_best)

            # ---- Console output ----
            best_marker = " (*)" if is_best else ""
            print(
                f"  Ep {epoch:04d}/{total_epochs-1:04d} | "
                f"TrainL={train_stats['loss']:.4f} | "
                f"ValL={val_stats['loss']:.4f} | "
                f"PSNR={val_stats['psnr']:.2f}dB | "
                f"SSIM={val_stats['ssim']:.4f} | "
                f"LR={train_stats['lr']:.2e} | "
                f"GN={train_stats['grad_norm']:.3f} | "
                f"{epoch_time:.1f}s"
                f"{best_marker}"
            )

            # ---- Early stopping ----
            epochs_no_improve = self.metric_tracker.epochs_since_improvement(epoch)
            if epochs_no_improve >= patience:
                print(f"\n  [EARLY STOP] No PSNR improvement for "
                      f"{patience} epochs. Stopping at epoch {epoch}.")
                break

        self.writer.close()
        print("\n" + "=" * 70)
        print(f"  Training complete.")
        print(f"  Best PSNR : {self.best_psnr:.4f} dB")
        print(f"  Best SSIM : {self.best_ssim:.6f}")
        print(f"  Best ckpt : {best_ckpt_path()}")
        print("=" * 70)
