"""
metrics/ssim_metric.py
=======================
AMSR-Net - Phase 5: SSIM Metric (from scratch)
------------------------------------------------
SSIM = Structural Similarity Index Measure (Wang et al., 2004).

Metric vs. Loss: what is different here?
-----------------------------------------
The SSIMLoss in Phase 4 was designed for backpropagation:
  - Returns a scalar (mean over batch and spatial dims)
  - Optimised for gradient stability

This SSIMLoss metric is designed for evaluation:
  - Returns per-image SSIM scores (B,) for proper epoch averaging
  - Optionally returns a spatial SSIM map for visualisation
  - Always detaches from the computation graph
  - Uses the exact Wang et al. parameters for comparability

The underlying Gaussian-windowed SSIM computation is shared in spirit
but reimplemented cleanly here for the evaluation context.

Standard SSIM Parameters (Wang et al., 2004)
---------------------------------------------
    window_size = 11
    sigma       = 1.5
    k1          = 0.01
    k2          = 0.03
    data_range  = 1.0  (images in [0, 1])
    C1 = (k1 * L)^2   = (0.01)^2 = 0.0001
    C2 = (k2 * L)^2   = (0.03)^2 = 0.0009

Note: Our SSIMLoss used data_range=2.0 because images are in [-1, +1].
This metric uses data_range=1.0 because images are DENORMALISED to [0, 1]
before being passed here. This matches the published SSIM formula.

Interpretation
--------------
    SSIM = 1.0   :  perfect structural match
    SSIM > 0.95  :  excellent (competitive SR results)
    SSIM > 0.90  :  good quality
    SSIM > 0.85  :  acceptable
    SSIM < 0.80  :  poor / blurry reconstruction

References
----------
- Wang, Z., et al. (2004). "Image quality assessment: from error visibility
  to structural similarity." IEEE TIP 13(4): 600-612.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    """Create a normalised 1D Gaussian kernel."""
    coords = torch.arange(window_size, dtype=torch.float32)
    coords = coords - (window_size - 1) / 2.0
    gauss  = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def _gaussian_2d(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a grouped depthwise 2D Gaussian kernel: (C, 1, W, W)."""
    g1d = _gaussian_1d(window_size, sigma)
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)   # outer product (W, W)
    g2d = g2d / g2d.sum()
    return g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()


class SSIMMetric(nn.Module):
    """
    SSIM evaluation metric with per-image scores and optional spatial maps.

    Computes the standard Wang et al. (2004) SSIM for images in [0, 1].

    Parameters
    ----------
    window_size : int    Gaussian window size (default 11).
    sigma       : float  Gaussian std (default 1.5).
    channels    : int    Image channel count (default 1).
    data_range  : float  Pixel value range (1.0 for [0,1] images).
    k1, k2      : float  Stability constants (Wang et al. defaults).
    reduction   : str    'mean' = scalar, 'none' = per-image (B,).
    """

    def __init__(
        self,
        window_size: int   = 11,
        sigma:       float = 1.5,
        channels:    int   = 1,
        data_range:  float = 1.0,
        k1:          float = 0.01,
        k2:          float = 0.03,
        reduction:   str   = "mean",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma       = sigma
        self.channels    = channels
        self.data_range  = data_range
        self.reduction   = reduction
        self.padding     = window_size // 2

        self.C1 = (k1 * data_range) ** 2   # 0.0001 for L=1
        self.C2 = (k2 * data_range) ** 2   # 0.0009 for L=1

        window = _gaussian_2d(window_size, sigma, channels)
        self.register_buffer("window", window)

    def _compute_ssim_map(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the local SSIM map.

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) in [0, 1].
        target : torch.Tensor  (B, C, H, W) in [0, 1].

        Returns
        -------
        torch.Tensor  (B, 1, H, W) SSIM map in [-1, 1].

        Mathematics
        -----------
        mu_x  = Gaussian_filter(pred)
        mu_y  = Gaussian_filter(target)

        Var(x) = Gaussian_filter(pred^2) - mu_x^2
        Var(y) = Gaussian_filter(target^2) - mu_y^2
        Cov(x,y) = Gaussian_filter(pred*target) - mu_x*mu_y

        SSIM = [(2*mu_x*mu_y + C1)*(2*Cov(x,y) + C2)]
               / [(mu_x^2 + mu_y^2 + C1)*(Var(x) + Var(y) + C2)]
        """
        C  = pred.shape[1]
        w  = self.window.to(pred.device, dtype=pred.dtype)

        # Expand kernel if runtime channel count differs from init
        if C != self.channels:
            w = _gaussian_2d(self.window_size, self.sigma, C).to(pred.device, dtype=pred.dtype)

        mu_x  = F.conv2d(pred,   w, padding=self.padding, groups=C)
        mu_y  = F.conv2d(target, w, padding=self.padding, groups=C)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy   = mu_x * mu_y

        sigma_x_sq  = F.conv2d(pred * pred,     w, padding=self.padding, groups=C) - mu_x_sq
        sigma_y_sq  = F.conv2d(target * target, w, padding=self.padding, groups=C) - mu_y_sq
        sigma_xy    = F.conv2d(pred * target,   w, padding=self.padding, groups=C) - mu_xy

        # Clamp variances to be non-negative (numerical precision issue)
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        numer = (2.0 * mu_xy + self.C1) * (2.0 * sigma_xy + self.C2)
        denom = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2)

        ssim_map = numer / (denom + 1e-10)   # (B, C, H, W)

        # Average over channels -> (B, 1, H, W)
        return ssim_map.mean(dim=1, keepdim=True)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SSIM score(s).

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) in [0, 1] -- DENORMALISED.
        target : torch.Tensor  (B, C, H, W) in [0, 1] -- DENORMALISED.

        Returns
        -------
        torch.Tensor
            Scalar mean SSIM   (if reduction='mean').
            Per-image SSIM (B,) (if reduction='none').
        """
        with torch.no_grad():
            ssim_map = self._compute_ssim_map(pred, target)   # (B, 1, H, W)

            # Per-image SSIM: mean over spatial dimensions
            ssim_per_image = ssim_map.mean(dim=[1, 2, 3])     # (B,)

            if self.reduction == "mean":
                return ssim_per_image.mean()
            return ssim_per_image

    def forward_with_map(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute SSIM and return both the scalar score and the spatial map.

        The spatial map is useful for visualising which regions the model
        is restoring well (high SSIM = green) vs. poorly (low SSIM = red).

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W).
        target : torch.Tensor  (B, C, H, W).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (scalar_ssim, ssim_map (B, 1, H, W))
        """
        with torch.no_grad():
            ssim_map       = self._compute_ssim_map(pred, target)
            scalar_ssim    = ssim_map.mean(dim=[1, 2, 3]).mean()
            return scalar_ssim, ssim_map

    def __repr__(self) -> str:
        return (
            f"SSIMMetric(window={self.window_size}, sigma={self.sigma}, "
            f"C1={self.C1:.6f}, C2={self.C2:.6f}, "
            f"range={self.data_range}, reduction='{self.reduction}')"
        )
