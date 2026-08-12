"""
losses/charbonnier.py
======================
AMSR-Net - Phase 4: Charbonnier Loss
--------------------------------------
Charbonnier loss is the primary pixel-wise fidelity loss for AMSR-Net.

Mathematical Definition
-----------------------
    L_charb(y_pred, y_gt) = mean( sqrt( (y_pred - y_gt)^2 + eps^2 ) )

where eps is a small constant (default 1e-3).

Why NOT MSE (L2)?
-----------------
Phase 1 noise analysis revealed:
    - Excess kurtosis = 4.45  (Gaussian = 0)
    - This means the noise has HEAVY TAILS (speckle noise)
    - Heavy tails produce outlier pixel values far from the mean

MSE penalises outliers QUADRATICALLY: loss = (x - y)^2
For a pixel 0.3 away from target: MSE penalty = 0.09
For a pixel 0.6 away from target: MSE penalty = 0.36  (4x more)

This causes the model to over-correct for outlier pixels and produce
BLURRY predictions (regression to the mean of many possible outputs).

Why Charbonnier?
----------------
Charbonnier = pseudo-Huber loss = smooth L1 approximation:
    - Near zero: behaves like L2 (smooth, stable gradient)
    - Far from zero: behaves like L1 (robust to outliers)
    - Unlike true L1: differentiable EVERYWHERE (no subgradient issues)

Comparison at delta=0.3:
    L2:         0.090
    L1:         0.300
    Charbonnier: ~0.300  (L1-like far from 0)

Comparison at delta=0.001:
    L2:          1e-6
    L1:          1e-3
    Charbonnier: ~1e-6  (L2-like near 0)

This makes Charbonnier ideal for images with mostly small errors (good
signal) but occasional large speckle outliers -- exactly our data.

References
----------
- Charbonnier et al. (1994): "Two deterministic half-quadratic regularization
  algorithms for computed imaging."
- Lai et al. (2017): Deep Laplacian Pyramid Networks -- used Charbonnier.
- Zamir et al. (2021): Restormer -- used Charbonnier as primary loss.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """
    Charbonnier (pseudo-Huber / smooth-L1) loss for image restoration.

    Formula
    -------
        L = mean( sqrt( (pred - target)^2 + eps^2 ) )

    The eps term:
      - Prevents the sqrt gradient from becoming infinite at zero
        (true L1 has undefined gradient at 0; this makes it C-infinity).
      - Controls the transition point between L2 and L1 behaviour:
        * |error| << eps  ->  L2-like  (quadratic, smooth)
        * |error| >> eps  ->  L1-like  (linear, robust)

    Parameters
    ----------
    eps        : float  Smoothing constant (default 1e-3).
                        Smaller eps = closer to true L1.
                        Larger  eps = closer to L2.
    reduction  : str    'mean' | 'sum' | 'none'
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.eps_sq = eps ** 2
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Charbonnier loss.

        Parameters
        ----------
        pred   : torch.Tensor  Model output, any shape, in [-1, +1].
        target : torch.Tensor  Ground truth, same shape as pred.

        Returns
        -------
        torch.Tensor  Scalar loss (if reduction='mean' or 'sum').
        """
        # Force FP32 to prevent eps_sq (1e-6) underflowing to 0 in AMP FP16,
        # which causes infinite gradients at sqrt(0)
        pred = pred.float()
        target = target.float()
        
        diff = pred - target                               # element-wise error
        loss = torch.sqrt(diff * diff + self.eps_sq)      # pseudo-Huber

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss   # 'none'

    def __repr__(self) -> str:
        return f"CharbonnierLoss(eps={self.eps}, reduction='{self.reduction}')"
