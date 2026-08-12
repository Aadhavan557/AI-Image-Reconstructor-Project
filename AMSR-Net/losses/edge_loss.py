"""
losses/edge_loss.py
====================
AMSR-Net - Phase 4: Edge / Gradient Loss (from scratch)
---------------------------------------------------------
Edge loss penalises errors in image gradients (edges), encouraging the
network to produce sharp boundaries rather than blurry transitions.

Why Edge Loss for Semiconductor Images?
----------------------------------------
Semiconductor wafer inspection images consist primarily of:
  - Sharp edges between circuit lines (line-space patterns)
  - High-contrast boundaries between conductor and insulator regions
  - Periodic fine structures with precise spatial frequencies

The Charbonnier loss treats all pixels equally. But perceptually and
metrologically, EDGE pixels are the most important: a slight blur at
a circuit edge can cause a false defect detection, dramatically impacting
yield prediction. Edge loss directly penalises these blurry boundaries.

Mathematical Definition
-----------------------
We compute image gradients using the Sobel operator:

    Sobel_x = [[-1, 0, +1],    Sobel_y = [[-1, -2, -1],
               [-2, 0, +2],                [ 0,  0,  0],
               [-1, 0, +1]]                [+1, +2, +1]]

Gradient magnitude at pixel (i,j):
    G = sqrt( G_x(i,j)^2 + G_y(i,j)^2 )

Edge loss = Charbonnier( G_pred, G_gt )
         = mean( sqrt( (G_pred - G_gt)^2 + eps^2 ) )

Why Charbonnier on gradients?
------------------------------
Using L2 on gradients would over-penalise large gradient differences
(sharp edges have very large gradient magnitudes). Charbonnier keeps
the loss robust to the heavy-tailed distribution of gradient errors
at true edge locations -- consistent with our Phase 1 finding.

Why Sobel over Laplacian?
--------------------------
The Laplacian is the second derivative -- it amplifies noise. The Sobel
operator is a first-order derivative filter, which gives a cleaner edge
signal for noisy images. Since our input NoisyLR has speckle, using the
Laplacian would amplify noise in the loss signal.

Implementation
--------------
The Sobel kernels are registered as fixed (non-trainable) buffers.
They are applied as a grouped depthwise convolution to support any
number of image channels.

References
----------
- Sobel & Feldman (1968): "A 3x3 Isotropic Gradient Operator for Image
  Processing." Presented at the Stanford AI Project.
- Johnson & Alahi (2016): Perceptual losses for real-time SR -- used
  gradient-based losses for sharpness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """
    Sobel-gradient edge loss for sharpness-preserving image restoration.

    Computes the Charbonnier loss between the gradient magnitudes of
    the predicted and ground-truth images.

    Parameters
    ----------
    channels : int    Number of image channels (default 1, grayscale).
    eps      : float  Charbonnier smoothing constant (default 1e-3).
    """

    def __init__(self, channels: int = 1, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps    = eps
        self.eps_sq = eps ** 2

        # ------------------------------------------------------------------
        # Sobel kernels -- fixed, not trained
        # ------------------------------------------------------------------
        # Horizontal gradient detector (strong response to vertical edges)
        sobel_x = torch.tensor(
            [[-1., 0., +1.],
             [-2., 0., +2.],
             [-1., 0., +1.]],
            dtype=torch.float32
        )
        # Vertical gradient detector (strong response to horizontal edges)
        sobel_y = torch.tensor(
            [[-1., -2., -1.],
             [ 0.,  0.,  0.],
             [+1., +2., +1.]],
            dtype=torch.float32
        )

        # Reshape for grouped depthwise conv: (C_out, C_in/groups, H, W)
        # For depthwise: out_channels = in_channels, groups = in_channels
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0).expand(channels, 1, 3, 3)

        # Register as buffers: move with .to(device) but not trained
        self.register_buffer("sobel_x", sobel_x.contiguous())
        self.register_buffer("sobel_y", sobel_y.contiguous())
        self.channels = channels

    def _gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel gradient magnitude using Sobel operators.

        Parameters
        ----------
        img : torch.Tensor  Input image (B, C, H, W).

        Returns
        -------
        torch.Tensor  Gradient magnitude (B, C, H, W).

        Formula
        -------
        G = sqrt( (Sobel_x * I)^2 + (Sobel_y * I)^2 )

        The sqrt is stabilised by eps_sq to avoid zero-gradient at
        flat regions (same reason as Charbonnier).
        """
        C = img.shape[1]

        # Cast kernels to match the input tensor's dtype (critical for AMP FP16)
        if C != self.channels:
            sx = self.sobel_x[:1].expand(C, 1, 3, 3).contiguous()
            sy = self.sobel_y[:1].expand(C, 1, 3, 3).contiguous()
        else:
            sx = self.sobel_x
            sy = self.sobel_y

        sx = sx.to(dtype=img.dtype, device=img.device)
        sy = sy.to(dtype=img.dtype, device=img.device)

        gx = F.conv2d(img, sx, padding=1, groups=C).float()   # (B, C, H, W) in FP32
        gy = F.conv2d(img, sy, padding=1, groups=C).float()   # (B, C, H, W) in FP32

        # Gradient magnitude: G = sqrt(gx^2 + gy^2 + eps^2)
        return torch.sqrt(gx * gx + gy * gy + self.eps_sq)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge loss = Charbonnier( grad_mag(pred), grad_mag(target) ).

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) model output in [-1, +1].
        target : torch.Tensor  (B, C, H, W) ground truth in [-1, +1].

        Returns
        -------
        torch.Tensor  Scalar edge loss.
        """
        # Force FP32 to prevent eps_sq (1e-6) underflowing to 0 in AMP FP16,
        # which causes infinite gradients at sqrt(0)
        pred = pred.float()
        target = target.float()
        
        grad_pred   = self._gradient_magnitude(pred)
        grad_target = self._gradient_magnitude(target)

        diff = grad_pred - grad_target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return loss.mean()

    def __repr__(self) -> str:
        return f"EdgeLoss(channels={self.channels}, eps={self.eps})"
