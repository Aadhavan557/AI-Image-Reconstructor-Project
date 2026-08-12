"""
metrics/metric_tracker.py
==========================
AMSR-Net - Phase 5: Metric Tracker
-------------------------------------
Accumulates per-batch PSNR and SSIM values during an epoch,
then computes the epoch-level mean with one call.

Why a tracker instead of calling metrics directly?
---------------------------------------------------
During a validation epoch, we iterate over many batches. If we simply
average the batch-level PSNR values, we get a biased estimate when
the last batch is smaller than the others (drop_last=False in val loader).

The tracker accumulates the SUM of per-image scores and the COUNT of
images, then computes the weighted mean at epoch end:

    epoch_psnr = sum(psnr_per_image) / n_images

This gives an UNBIASED estimate of the per-image epoch mean regardless
of batch size variation.

Additionally, the tracker records the best epoch values and detects
improvement for early stopping, without the training loop needing to
manage state.

Design
------
This is a lightweight pure-Python class (no nn.Module) because it
accumulates Python floats, not tensors. This avoids GPU memory
accumulation across the entire validation set.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from metrics.psnr        import PSNRMetric
from metrics.ssim_metric import SSIMMetric


class MetricTracker:
    """
    Epoch-level accumulator for PSNR and SSIM metrics.

    Usage
    -----
    tracker = MetricTracker(device=device)
    tracker.reset()

    for batch in val_loader:
        pred, gt = model(batch["lr"]), batch["gt"]
        # Denormalise from [-1,+1] to [0,1] before computing metrics
        pred_01 = (pred * 0.5 + 0.5).clamp(0, 1)
        gt_01   = (gt   * 0.5 + 0.5).clamp(0, 1)
        tracker.update(pred_01, gt_01)

    epoch_stats = tracker.compute()
    # -> {"psnr": 32.5, "ssim": 0.921, "n_images": 320}

    Parameters
    ----------
    device : torch.device  Device for metric computations.
    """

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

        self._psnr_fn = PSNRMetric(data_range=1.0, reduction="none").to(self.device)
        self._ssim_fn = SSIMMetric(
            channels=1, data_range=1.0, reduction="none"
        ).to(self.device)

        self._psnr_sum: float = 0.0
        self._ssim_sum: float = 0.0
        self._n_images: int   = 0

        # Best-ever tracking (for early stopping and checkpointing)
        self.best_psnr: float = -1.0
        self.best_ssim: float = -1.0
        self.best_epoch: int  = -1

    def reset(self) -> None:
        """Reset accumulators for a new epoch. Call at epoch start."""
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._n_images = 0

    def update(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Accumulate metric values for a single batch.

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) in [0, 1] -- MUST be denormalised.
        target : torch.Tensor  (B, C, H, W) in [0, 1] -- MUST be denormalised.

        Returns
        -------
        Dict  Batch-level {"psnr": float, "ssim": float} for live logging.
        """
        pred   = pred.to(self.device)
        target = target.to(self.device)

        # Per-image metrics, shape (B,)
        psnr_batch = self._psnr_fn(pred, target)   # (B,) or scalar
        ssim_batch = self._ssim_fn(pred, target)   # (B,) or scalar

        # Handle reduction='none' -> (B,) tensor
        if psnr_batch.ndim == 0:
            psnr_batch = psnr_batch.unsqueeze(0)
        if ssim_batch.ndim == 0:
            ssim_batch = ssim_batch.unsqueeze(0)

        b = psnr_batch.shape[0]

        self._psnr_sum += psnr_batch.sum().item()
        self._ssim_sum += ssim_batch.sum().item()
        self._n_images += b

        return {
            "psnr": psnr_batch.mean().item(),
            "ssim": ssim_batch.mean().item(),
        }

    def compute(self) -> Dict[str, float]:
        """
        Compute epoch-level mean metrics from accumulated sums.

        Returns
        -------
        Dict with keys:
            "psnr"     : float  Mean PSNR in dB.
            "ssim"     : float  Mean SSIM in [0, 1].
            "n_images" : int    Total images processed.
        """
        if self._n_images == 0:
            return {"psnr": 0.0, "ssim": 0.0, "n_images": 0}

        return {
            "psnr":     self._psnr_sum / self._n_images,
            "ssim":     self._ssim_sum / self._n_images,
            "n_images": self._n_images,
        }

    def update_best(self, epoch: int) -> Dict[str, bool]:
        """
        Check if current epoch is a new best, and update best records.

        Call AFTER compute() at the end of each validation epoch.

        Parameters
        ----------
        epoch : int  Current epoch index.

        Returns
        -------
        Dict with "psnr_improved" and "ssim_improved" boolean flags.
        Used by trainer to decide whether to save a checkpoint.
        """
        stats = self.compute()
        improved = {"psnr_improved": False, "ssim_improved": False}

        if stats["psnr"] > self.best_psnr:
            self.best_psnr = stats["psnr"]
            improved["psnr_improved"] = True

        if stats["ssim"] > self.best_ssim:
            self.best_ssim = stats["ssim"]
            improved["ssim_improved"] = True

        if improved["psnr_improved"] or improved["ssim_improved"]:
            self.best_epoch = epoch

        return improved

    def epochs_since_improvement(self, current_epoch: int) -> int:
        """
        Return number of epochs since last PSNR improvement.
        Used for early stopping decisions.

        Parameters
        ----------
        current_epoch : int  Current epoch index.

        Returns
        -------
        int  Epochs since the best epoch was recorded.
        """
        if self.best_epoch < 0:
            return 0
        return current_epoch - self.best_epoch

    def summary(self) -> str:
        """Return a formatted string of current epoch stats."""
        stats = self.compute()
        return (
            f"PSNR={stats['psnr']:.4f} dB  "
            f"SSIM={stats['ssim']:.6f}  "
            f"[{stats['n_images']} images]"
        )

    def __repr__(self) -> str:
        return (
            f"MetricTracker("
            f"best_psnr={self.best_psnr:.2f} dB, "
            f"best_ssim={self.best_ssim:.4f}, "
            f"best_epoch={self.best_epoch})"
        )
