"""
models/baseline_cnn.py
======================
AMSR-Net - Phase 3: Baseline CNN
----------------------------------
Architecture: Residual Channel Attention Network for Joint Denoising + 2x SR
-----------------------------------------------------------------------------

Design Rationale
----------------
This baseline is inspired by EDSR (Enhanced Deep SR, Lim et al. 2017) with
two critical additions justified by Phase 1 findings:

1. Channel Attention (SE block, Hu et al. 2018)
   Phase 1 showed speckle noise with kurtosis=4.45 (heavy tails).
   Channel attention lets the network learn WHICH feature channels
   carry structural signal vs. which carry noise, improving the
   signal-to-noise ratio in feature space before upsampling.

2. No BatchNorm in residual blocks (EDSR's key insight)
   BatchNorm normalises feature statistics which REMOVES range
   flexibility. For restoration tasks where input range is abnormal
   (NoisyLR goes to ±1.4), removing BN lets the network preserve
   the full dynamic range of features.

Architecture Overview
---------------------

  Input: (B, 1, 64, 64)   -- NoisyLR, normalised to [-1.4, +1.4]
         |
  [ShallowFeatureExtractor]    3x3 Conv  ->  dim channels
         |
  [ResidualGroup x num_blocks]
    Each block:
      Conv(3x3) -> ReLU -> Conv(3x3) -> ChannelAttention -> residual
         |
  [1x1 Conv]                  collapse to dim channels
         |
  [global residual add]       feature + shallow_feature
         |
  [PixelShuffleUpsample]      2x: Conv(3x3) -> PixelShuffle(2)
         |
  [ReconstructionHead]        Conv(3x3) -> tanh
         |
  Output: (B, 1, 128, 128)  -- restored GT, in [-1, +1]

Why PixelShuffle for upsampling?
---------------------------------
PixelShuffle (sub-pixel convolution, Shi et al. 2016) avoids the
checkerboard artefacts produced by transposed convolutions when the
kernel stride does not evenly divide. It rearranges depth channels
into spatial dimensions: (B, C*r^2, H, W) -> (B, C, H*r, W*r).
For semiconductor images with fine periodic structures (circuit lines),
avoiding checkerboard artefacts is essential.

Why tanh at the output?
------------------------
The network is trained to predict images in [-1, +1] (our normalisation
from Phase 2). tanh enforces this range asymptotically, preventing
extreme outlier predictions that would destabilise early training.

Parameters (with cfg defaults: dim=64, num_blocks=8)
------------------------------------------------------
Estimated parameters: ~1.2M (suitable as a fast baseline).
The full AMSR-Net hybrid will be ~10-50x larger.
"""

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg


# ===========================================================================
# Building Blocks
# ===========================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention block (Hu et al., CVPR 2018).

    Operation
    ---------
    1. Global average pool: (B, C, H, W) -> (B, C, 1, 1)
    2. FC squeeze: C -> C // reduction
    3. ReLU
    4. FC excite: C // reduction -> C
    5. Sigmoid gate: scale each channel of the feature map

    Mathematical formulation
    ------------------------
    Let F in R^(C x H x W) be the input feature map.

        z_c = (1/HW) * sum_{i,j} F_c(i,j)         [squeeze]
        s   = sigmoid( W2 * relu( W1 * z ) )        [excite]
        F'  = s * F                                  [scale]

    where W1 in R^(C/r x C), W2 in R^(C x C/r), r = reduction ratio.

    The sigmoid gate s_c in (0, 1) acts as a soft selector, suppressing
    channels dominated by noise and amplifying channels with structural
    signal -- exactly what is needed given our speckle noise model.

    Parameters
    ----------
    channels  : int  Number of input/output channels.
    reduction : int  Bottleneck reduction ratio (default: 8).
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)   # guard against tiny channel counts
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)     # (B, C)
        y = self.fc(y).view(b, c, 1, 1)     # (B, C, 1, 1)
        return x * y                          # channel-wise scale


class ResidualBlock(nn.Module):
    """
    Residual block WITHOUT batch normalisation (EDSR-style).

    Structure
    ---------
        Input
          |
        Conv(3x3, pad=1)
          |
        ReLU (inplace)
          |
        Conv(3x3, pad=1)
          |
        ChannelAttention
          |
        * residual_scale
          |
        + Input
          |
        Output

    Why no BatchNorm?
    -----------------
    Lim et al. (EDSR, 2017) showed that removing BN in SR residual blocks
    improves PSNR. The intuition: BN constrains the feature distribution
    to zero mean / unit variance, which discards range information that
    is critical when the input has a non-standard dynamic range (as our
    NoisyLR does, reaching +-1.4 after normalisation).

    Residual Scaling
    ----------------
    We apply a scalar multiplier (default 0.1) to the residual branch.
    This prevents residuals from exploding early in training when
    channels = 64 and num_blocks = 8 (empirically validated in EDSR).

    Parameters
    ----------
    channels : int    Number of feature channels.
    reduction : int   SE reduction ratio.
    res_scale : float Residual branch scaling factor.
    """

    def __init__(
        self,
        channels:  int   = 64,
        reduction: int   = 8,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            ChannelAttention(channels, reduction=reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.body(x)


class PixelShuffleUpsample(nn.Module):
    """
    2× upsampling via sub-pixel convolution (PixelShuffle).

    Operation
    ---------
    1. Conv2d: C -> C * scale^2  (expand channels to hold all output pixels)
    2. PixelShuffle(scale): rearrange (B, C*r^2, H, W) -> (B, C, H*r, W*r)

    Why PixelShuffle over transposed convolution?
    ---------------------------------------------
    Transposed convolutions with stride > 1 are prone to checkerboard
    artefacts (Odena et al., 2016) because zero-padding between input
    pixels creates uneven contribution at output pixels. PixelShuffle
    avoids this entirely by operating in the original (non-zero-padded)
    resolution and only expanding dimensionality at the very end.

    For semiconductor images with fine line patterns (period ≈ 2-8 pixels),
    checkerboard artefacts at the scale of 2-4 pixels would be
    indistinguishable from circuit features -- a critical failure mode.

    Parameters
    ----------
    in_channels : int  Input channel count.
    scale       : int  Upscaling factor (default 2).
    """

    def __init__(self, in_channels: int, scale: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * (scale ** 2), 3, padding=1, bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.conv(x))


# ===========================================================================
# Baseline CNN (full model)
# ===========================================================================

class BaselineCNN(nn.Module):
    """
    Residual Channel Attention CNN for joint denoising + 2x super-resolution.

    This is the Phase 3 baseline model. It is designed to:
    - Establish a strong performance floor before the hybrid AMSR-Net.
    - Validate the training loop, loss functions, and metrics (Phases 8-9).
    - Provide a model whose weights can be used to warm-start the AMSR-Net.

    Input  : (B, in_channels, H,    W   )  NoisyLR in [-1.4, +1.4]
    Output : (B, in_channels, H*2,  W*2 )  Restored GT in [-1, +1]

    Parameters
    ----------
    in_channels : int   Number of image channels (1 for grayscale).
    dim         : int   Internal feature channel dimension.
    num_blocks  : int   Number of residual blocks.
    scale       : int   Super-resolution upscaling factor.
    reduction   : int   SE channel attention reduction ratio.
    res_scale   : float Residual branch scalar multiplier.
    """

    def __init__(
        self,
        in_channels: int   = cfg.CHANNELS,
        dim:         int   = cfg.BASELINE_FEATURES,
        num_blocks:  int   = cfg.BASELINE_NUM_BLOCKS,
        scale:       int   = cfg.SCALE,
        reduction:   int   = 8,
        res_scale:   float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale

        # ------------------------------------------------------------------
        # Stage 1: Shallow feature extraction
        # ------------------------------------------------------------------
        # A single 3x3 conv maps the raw 1-channel input to `dim` features.
        # Using a large receptive field here would be wasteful -- the first
        # layer should capture local texture, not global context.
        self.shallow = nn.Conv2d(in_channels, dim, 3, padding=1, bias=True)

        # ------------------------------------------------------------------
        # Stage 2: Deep feature extraction (stack of residual blocks)
        # ------------------------------------------------------------------
        self.body = nn.Sequential(
            *[ResidualBlock(dim, reduction=reduction, res_scale=res_scale)
              for _ in range(num_blocks)]
        )

        # 1x1 conv after the stack (projects features, no spatial mixing)
        self.body_end = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

        # ------------------------------------------------------------------
        # Stage 3: Global residual connection
        # The global residual is the KEY architectural trick in EDSR:
        # Instead of learning F(x), the body learns the residual R(x) = F(x)-x.
        # The skip connection is: output_features = body(x) + x
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Stage 4: 2x PixelShuffle upsampling
        # ------------------------------------------------------------------
        self.upsample = PixelShuffleUpsample(dim, scale=scale)

        # ------------------------------------------------------------------
        # Stage 5: Reconstruction head
        # ------------------------------------------------------------------
        # 3x3 conv maps `dim` features -> 1 output channel.
        # We removed Tanh to allow the network to learn the full residual.
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, in_channels, 3, padding=1, bias=True),
        )

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Kaiming (He) initialisation for Conv layers.

        Why Kaiming?
        ------------
        Kaiming Normal initialisation for all Conv/Linear layers.
        Bias is initialised to zero (standard practice).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                                        
        # Zero-initialize the final reconstruction layer 
        # so the network starts as an exact identity mapping (preventing dead residual plateau)
        if isinstance(self.head[-1], nn.Conv2d):
            nn.init.zeros_(self.head[-1].weight)
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input NoisyLR tensor, shape (B, 1, H, W), normalised to ~[-1.4, +1.4].

        Returns
        -------
        torch.Tensor
            Restored HR tensor, shape (B, 1, H*scale, W*scale), in [-1, +1].
        """
        # Stage 1: shallow feature extraction
        shallow = self.shallow(x)          # (B, dim, H, W)

        # Stage 2: deep feature extraction
        deep = self.body(shallow)          # (B, dim, H, W)
        deep = self.body_end(deep)         # (B, dim, H, W)

        # Stage 3: global residual
        feat = deep + shallow              # (B, dim, H, W)

        # Stage 4: upsample
        feat = self.upsample(feat)         # (B, dim, H*2, W*2)

        # Stage 5: reconstruct
        out = self.head(feat)              # (B, 1, H*2, W*2)
        
        # Global image-space residual connection
        base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return out + base

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_params = self.count_parameters()
        return (
            f"BaselineCNN(\n"
            f"  dim        = {self.shallow.out_channels}\n"
            f"  num_blocks = {len([m for m in self.body if isinstance(m, ResidualBlock)])}\n"
            f"  scale      = {self.scale}x\n"
            f"  params     = {n_params:,}  ({n_params/1e6:.2f}M)\n"
            f")"
        )
