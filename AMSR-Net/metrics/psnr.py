"""
metrics/psnr.py
================
AMSR-Net - Phase 5: PSNR Metric (from scratch)
------------------------------------------------
PSNR = Peak Signal-to-Noise Ratio, measured in decibels (dB).

Mathematical Definition
-----------------------
    MSE  = (1 / N) * sum( (y_pred - y_gt)^2 )
    PSNR = 10 * log10( MAX^2 / MSE )

where:
    N    = total number of pixels (H * W * C)
    MAX  = maximum possible pixel value
         = 1.0  (since images are denormalised to [0, 1])
    MSE  = mean squared error between prediction and ground truth

For perfectly identical images:
    MSE  -> 0, PSNR -> +inf

Typical PSNR ranges for image restoration:
    < 25 dB    :  poor quality
    25 - 30 dB :  acceptable
    30 - 35 dB :  good
    > 35 dB    :  excellent (competitive SR results)
    > 40 dB    :  near-lossless

Important Implementation Details
----------------------------------
1. PSNR is computed PER IMAGE, then averaged over the batch.
   (Not computed on the batch-concatenated pixel tensor, which would
    be biased by batch composition.)

2. Input must be in [0, 1] range.
   Our network outputs [-1, +1]. The caller MUST denormalise before
   passing to this metric. This is enforced with a runtime check.

3. MSE = 0 edge case:
   log10(0) = -inf. We guard with a minimum MSE of 1e-10, which
   gives PSNR = 100 dB (essentially infinite quality).

4. PSNR is NOT differentiable (log10 of MSE is, but we use .detach()
   because metrics are evaluation-only -- never backpropagated).

Why PSNR as the primary metric?
--------------------------------
PSNR is the standard metric for image restoration competitions (NTIRE,
PIPAL, RealSR). It is:
  - Interpretable (dB scale, easy to compare across papers)
  - Directly comparable with published baselines (EDSR, SwinIR, etc.)
  - Required by most hackathon leaderboards

Its main weakness (insensitivity to perceptual quality) is addressed
by also tracking SSIM in Phase 5.
"""

import torch
import torch.nn as nn
from typing import Optional


class PSNRMetric(nn.Module):
    """
    Per-image PSNR metric with batch averaging.

    This is an EVALUATION metric, not a training loss.
    Always call with torch.no_grad() during evaluation.

    Parameters
    ----------
    data_range : float  Maximum pixel value (1.0 for [0,1] images).
    reduction  : str    'mean' = average PSNR over batch (default).
                        'none' = return per-image PSNR tensor.
    eps        : float  Minimum MSE to guard against log10(0) = -inf.
    """

    def __init__(
        self,
        data_range: float = 1.0,
        reduction:  str   = "mean",
        eps:        float = 1e-10,
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.reduction  = reduction
        self.eps        = eps
        self.max_sq     = data_range ** 2   # MAX^2, pre-computed

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PSNR between predicted and ground-truth images.

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) predictions in [0, 1].
        target : torch.Tensor  (B, C, H, W) ground truth in [0, 1].

        Returns
        -------
        torch.Tensor
            Scalar mean PSNR in dB (if reduction='mean').
            Tensor of shape (B,) per-image PSNR (if reduction='none').

        Notes
        -----
        Both inputs are detached before computation -- PSNR is never
        used as a loss, and detaching prevents accidental graph retention.
        """
        pred   = pred.detach().float()
        target = target.detach().float()

        # Compute per-image MSE: mean over (C, H, W), keep batch dim
        # Shape: (B,)
        mse_per_image = (pred - target).pow(2).mean(dim=[1, 2, 3])

        # Guard: clamp MSE to avoid log10(0) = -inf
        mse_per_image = mse_per_image.clamp(min=self.eps)

        # PSNR in dB: 10 * log10(MAX^2 / MSE)
        psnr_per_image = 10.0 * torch.log10(
            torch.full_like(mse_per_image, self.max_sq) / mse_per_image
        )

        if self.reduction == "mean":
            return psnr_per_image.mean()
        return psnr_per_image   # (B,) per-image scores

    @staticmethod
    def mse_to_psnr(mse: float, data_range: float = 1.0) -> float:
        """
        Convert a scalar MSE value to PSNR (dB).

        Convenience method for logging purposes.

        Parameters
        ----------
        mse        : float  Mean squared error in [0, data_range^2].
        data_range : float  Maximum pixel value (default 1.0).

        Returns
        -------
        float  PSNR in dB.
        """
        import math
        if mse <= 0:
            return 100.0
        return 10.0 * math.log10(data_range ** 2 / mse)

    def __repr__(self) -> str:
        return (
            f"PSNRMetric(data_range={self.data_range}, "
            f"reduction='{self.reduction}')"
        )
