"""
losses/composite_loss.py
========================
AMSR-Net - Phase 4: Composite Loss Function
---------------------------------------------
Combines Charbonnier, SSIM, and Edge losses with configurable weights
into a single differentiable objective for training AMSR-Net.

Loss Composition
----------------
    L_total = w_charb * L_charb
            + w_ssim  * L_ssim
            + w_edge  * L_edge

Why a composite loss?
---------------------
Each individual loss captures a different aspect of restoration quality:

    Loss         | What it measures         | What it optimises
    -------------|--------------------------|-----------------------------
    Charbonnier  | Pixel-level fidelity     | Overall brightness/colour
    SSIM         | Local structural pattern | Perceptual texture + contrast
    Edge         | Sharpness of boundaries  | Circuit edge precision

No single loss covers all three. The composite loss forces the network to
simultaneously satisfy all three objectives, producing outputs that are:
  - Bright/dark in the right places (Charbonnier)
  - Structurally similar (SSIM)
  - Sharply bounded (Edge)

Weight Justification
--------------------
Default weights from cfg:
    w_charb = 1.0   (primary pixel fidelity, full weight)
    w_ssim  = 0.2   (structural, secondary -- SSIM loss scale is ~0.0-1.0)
    w_edge  = 0.1   (sharpness, tertiary -- gradient magnitudes are ~10x larger)

The weights are chosen so that at the start of training all three loss
terms contribute a similar ORDER OF MAGNITUDE to the total gradient.
If w_edge were 1.0, the large Sobel gradient magnitudes would dominate
and the network would become a pure edge detector.

Note: These are starting defaults. The optimal weights may require
tuning via ablation study (Phase 9).

References
----------
- Zhang et al. (2018): LPIPS -- demonstrated perceptual losses outperform
  MSE for image restoration.
- Zhao et al. (2017): "Loss functions for image restoration with neural
  networks." -- demonstrated SSIM + L1 composite loss improves PSNR/SSIM.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from losses.charbonnier import CharbonnierLoss
from losses.ssim_loss    import SSIMLoss
from losses.edge_loss    import EdgeLoss


class CompositeLoss(nn.Module):
    """
    Weighted composite loss: Charbonnier + SSIM + Edge.

    Parameters
    ----------
    w_charb    : float  Weight for Charbonnier loss  (default cfg.LOSS_L1_W
                        -- note: config uses L1_W as the Charbonnier weight
                        because Charbonnier is our L1-like primary loss).
    w_ssim     : float  Weight for SSIM loss          (default cfg.LOSS_SSIM_W).
    w_edge     : float  Weight for Edge loss           (default cfg.LOSS_EDGE_W).
    channels   : int    Image channels                 (default cfg.CHANNELS).
    charb_eps  : float  Charbonnier smoothing constant.
    ssim_win   : int    SSIM Gaussian window size.
    data_range : float  Pixel value range (2.0 for [-1,+1] normalisation).
    """

    def __init__(
        self,
        w_charb:    float = cfg.LOSS_CHARBONNIER_W,
        w_ssim:     float = cfg.LOSS_SSIM_W,
        w_edge:     float = cfg.LOSS_EDGE_W,
        channels:   int   = cfg.CHANNELS,
        charb_eps:  float = 1e-3,
        ssim_win:   int   = 11,
        data_range: float = 2.0,
    ) -> None:
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim  = w_ssim
        self.w_edge  = w_edge

        self.charb_loss = CharbonnierLoss(eps=charb_eps)
        self.ssim_loss  = SSIMLoss(
            window_size = ssim_win,
            channels    = channels,
            data_range  = data_range,
        )
        self.edge_loss  = EdgeLoss(channels=channels, eps=charb_eps)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted composite loss.

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W) model output in [-1, +1].
        target : torch.Tensor  (B, C, H, W) ground truth in [-1, +1].

        Returns
        -------
        torch.Tensor  Scalar total loss.
        """
        l_charb = self.charb_loss(pred, target)
        l_ssim  = self.ssim_loss(pred, target)
        l_edge  = self.edge_loss(pred, target)

        total = (
            self.w_charb * l_charb
            + self.w_ssim  * l_ssim
            + self.w_edge  * l_edge
        )
        return total

    def forward_detailed(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute and return all individual loss components.

        Used for TensorBoard logging -- records each loss component
        separately so we can diagnose which component is improving.

        Parameters
        ----------
        pred   : torch.Tensor  (B, C, H, W).
        target : torch.Tensor  (B, C, H, W).

        Returns
        -------
        Dict with keys:
            "total"     : total weighted loss
            "charb"     : raw Charbonnier loss
            "ssim"      : raw SSIM loss (1 - SSIM_score)
            "edge"      : raw edge loss
            "ssim_score": raw SSIM score (for metric logging, NOT the loss)
        """
        l_charb = self.charb_loss(pred, target)
        l_ssim  = self.ssim_loss(pred, target)
        l_edge  = self.edge_loss(pred, target)

        total = (
            self.w_charb * l_charb
            + self.w_ssim  * l_ssim
            + self.w_edge  * l_edge
        )

        return {
            "total":      total,
            "charb":      l_charb,
            "ssim":       l_ssim,
            "edge":       l_edge,
            "ssim_score": self.ssim_loss.ssim_score(pred, target),
        }

    def __repr__(self) -> str:
        return (
            f"CompositeLoss(\n"
            f"  Charbonnier  weight={self.w_charb}  {self.charb_loss}\n"
            f"  SSIM         weight={self.w_ssim}   {self.ssim_loss}\n"
            f"  Edge         weight={self.w_edge}   {self.edge_loss}\n"
            f")"
        )
