"""
losses/ssim_loss.py
====================
AMSR-Net - Phase 4: SSIM Loss (from scratch)
----------------------------------------------
SSIM = Structural Similarity Index Measure (Wang et al., 2004).

Why SSIM as a loss?
-------------------
Charbonnier loss is pixel-wise -- it has no notion of local structure.
Two images can have the same Charbonnier loss but very different perceptual
quality (e.g., a spatially shifted prediction looks visually wrong but
pixel-wise loss may be similar).

SSIM quantifies perceptual similarity by measuring three components:
    1. Luminance (brightness)
    2. Contrast (variance)
    3. Structure (correlation of local patterns)

For semiconductor images:
    - "Structure" = circuit line patterns, feature edges, periodic gratings
    - These are exactly the signals the inspection engineer cares about
    - SSIM loss directly optimises for structural fidelity

Mathematical Definition
-----------------------
For two image patches x and y:

    mu_x  = E[x]          (local mean, computed via Gaussian window)
    mu_y  = E[y]
    sig_x = Var(x)^0.5    (local standard deviation)
    sig_y = Var(y)^0.5
    sig_xy = Cov(x, y)    (local cross-covariance)

    SSIM(x, y) = [ (2*mu_x*mu_y + C1) * (2*sig_xy + C2) ]
                 / [ (mu_x^2 + mu_y^2 + C1) * (sig_x^2 + sig_y^2 + C2) ]

where:
    C1 = (k1 * L)^2    stabilises the luminance term
    C2 = (k2 * L)^2    stabilises the contrast term
    L  = 1.0           data range (images are in [-1, +1] but relative range is 1)
    k1 = 0.01, k2 = 0.03  (standard constants from Wang et al.)

SSIM is in [-1, +1]. SSIM=1 means perfect structural match.

Loss = 1 - SSIM  (so minimising loss maximises structural similarity).

Implementation Notes
--------------------
- Gaussian window is implemented as a depthwise separable 2D conv.
  This is more efficient than a naive 2D Gaussian kernel for large images.
- The window std is 1.5 (standard from Wang et al.) and size is 11x11.
- We compute SSIM on each image in the batch and average.
- For multi-channel: mean over channels then over batch.

References
----------
- Wang et al. (2004): "Image quality assessment: from error visibility to
  structural similarity." IEEE TIP.
- Zhao et al. (2017): "Loss functions for image restoration with neural
  networks." IEEE TCI -- demonstrated SSIM loss improves perceptual quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel normalised to sum to 1.

    Parameters
    ----------
    window_size : int    Number of elements in the kernel.
    sigma       : float  Standard deviation of the Gaussian.

    Returns
    -------
    torch.Tensor  Shape (window_size,), normalised.

    Mathematics
    -----------
    g(x) = exp( -x^2 / (2 * sigma^2) )
    The kernel is evaluated at integer positions centred at 0:
    x in {-(N-1)/2, ..., 0, ..., +(N-1)/2}
    Then normalised: g_norm = g / sum(g)
    """
    coords = torch.arange(window_size, dtype=torch.float32)
    coords = coords - (window_size - 1) / 2.0          # centre at 0
    gauss  = torch.exp(-0.5 * (coords / sigma) ** 2)
    return gauss / gauss.sum()


def _build_gaussian_window(
    window_size: int, sigma: float, channels: int
) -> torch.Tensor:
    """
    Build a 2D separable Gaussian kernel for depthwise convolution.

    The 2D Gaussian is constructed as the outer product of two 1D Gaussians.
    Using separable convolution (two 1D convolutions) is O(N) vs O(N^2)
    per pixel, but here we construct the full 2D kernel for simplicity
    and use it with a single grouped conv2d call.

    Parameters
    ----------
    window_size : int      Kernel side length (e.g. 11).
    sigma       : float    Gaussian std (e.g. 1.5).
    channels    : int      Number of image channels (for grouped conv).

    Returns
    -------
    torch.Tensor  Shape (channels, 1, window_size, window_size).
    """
    _1d = _gaussian_kernel_1d(window_size, sigma)
    _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)          # outer product: (W, W)
    _2d = _2d / _2d.sum()                               # normalise to sum=1
    kernel = _2d.unsqueeze(0).unsqueeze(0)              # (1, 1, W, W)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    return kernel.contiguous()


class SSIMLoss(nn.Module):
    """
    SSIM-based perceptual loss for image restoration.

    Loss = 1 - SSIM(pred, target)

    A lower loss means higher structural similarity.

    Parameters
    ----------
    window_size : int    Local patch size for statistics (default 11).
    sigma       : float  Gaussian window std (default 1.5).
    channels    : int    Image channels (default 1 for grayscale).
    data_range  : float  Max pixel value range.
                         Our images are in [-1, +1] so range = 2.0.
    k1          : float  Luminance stabilisation constant (default 0.01).
    k2          : float  Contrast stabilisation constant (default 0.03).
    reduction   : str    'mean' | 'none' (over batch).
    """

    def __init__(
        self,
        window_size: int   = 11,
        sigma:       float = 1.5,
        channels:    int   = 1,
        data_range:  float = 2.0,
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

        # Stability constants (Wang et al., 2004)
        self.C1 = (k1 * data_range) ** 2   # stabilises luminance: ~0.0004
        self.C2 = (k2 * data_range) ** 2   # stabilises contrast:  ~0.0036

        # Register Gaussian window as a buffer (not a parameter -- not trained)
        window = _build_gaussian_window(window_size, sigma, channels)
        self.register_buffer("window", window)

    def _ssim_map(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM for each spatial position (returns a map, not a scalar).

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W)
        target : torch.Tensor  (B, C, H, W)

        Returns
        -------
        torch.Tensor  (B, 1, H', W') SSIM map (H'=H-pad, W'=W-pad)
        """
        C  = pred.shape[1]
        # Cast to match AMP FP16 if autocast is active
        w  = self.window.to(device=pred.device, dtype=pred.dtype)

        # Local means via depthwise (grouped) Gaussian convolution
        mu_x  = F.conv2d(pred,   w, padding=self.padding, groups=C)
        mu_y  = F.conv2d(target, w, padding=self.padding, groups=C)

        mu_x_sq  = mu_x * mu_x
        mu_y_sq  = mu_y * mu_y
        mu_xy    = mu_x * mu_y

        # Local variances and covariance
        # Var(X) = E[X^2] - E[X]^2
        sig_x_sq  = F.conv2d(pred   * pred,   w, padding=self.padding, groups=C) - mu_x_sq
        sig_y_sq  = F.conv2d(target * target, w, padding=self.padding, groups=C) - mu_y_sq
        sig_xy    = F.conv2d(pred   * target, w, padding=self.padding, groups=C) - mu_xy

        # SSIM formula (Wang et al., eq. 13)
        numerator   = (2.0 * mu_xy  + self.C1) * (2.0 * sig_xy  + self.C2)
        denominator = (mu_x_sq + mu_y_sq + self.C1) * (sig_x_sq + sig_y_sq + self.C2)

        ssim_map = numerator / (denominator + 1e-8)   # (B, C, H, W)

        # Average over channels -> (B, 1, H, W)
        return ssim_map.mean(dim=1, keepdim=True)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SSIM loss = 1 - mean(SSIM_map).

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) predictions in [-1, +1].
        target : torch.Tensor  (B, C, H, W) ground truth in [-1, +1].

        Returns
        -------
        torch.Tensor  Scalar loss in [0, 2].
                      (SSIM in [-1,+1] -> loss=1-SSIM in [0,2];
                       in practice SSIM > 0 so loss in [0,1])
        """
        ssim_map = self._ssim_map(pred, target)          # (B, 1, H, W)

        if self.reduction == "mean":
            return 1.0 - ssim_map.mean()
        return 1.0 - ssim_map.mean(dim=[1, 2, 3])        # per-image

    def ssim_score(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the raw SSIM score (not the loss).
        Useful for logging during training.

        Returns
        -------
        torch.Tensor  Scalar SSIM in approximately [0, 1].
        """
        return self._ssim_map(pred, target).mean()

    def __repr__(self) -> str:
        return (
            f"SSIMLoss(window={self.window_size}, sigma={self.sigma}, "
            f"C1={self.C1:.6f}, C2={self.C2:.6f}, range={self.data_range})"
        )
