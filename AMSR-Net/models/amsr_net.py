"""
models/amsr_net.py
===================
AMSR-Net - Phase 7: Adaptive Multi-Expert Semiconductor Restoration Network
---------------------------------------------------------------------------
The full hybrid model combining CNN, Restormer, and Swin Transformer
blocks into a unified image restoration architecture.

Architecture Design Rationale
-------------------------------

The AMSR-Net name reflects its "multi-expert" nature:
  - Expert 1 (CNN ResBlocks):    Local feature extraction, translation equivariance
  - Expert 2 (Restormer MDTA):   Global channel attention for denoising
  - Expert 3 (Swin W/SW-MSA):    Local spatial attention for SR edge sharpening

Each expert handles a different aspect of the joint denoising + 2x SR task.
The key insight is that NEITHER a pure CNN NOR a pure transformer is optimal:
  - Pure CNN: fast, but limited receptive field for global noise patterns
  - Pure Transformer: large receptive field, but expensive and unstable to train
  - Hybrid: combines the best of both worlds at manageable compute cost

Architecture Overview
----------------------

    Input: (B, 1, H, W)     NoisyLR in [-1.4, +1.4]
        |
    [Stage 0: Shallow Feature Extraction]
    ShallowConv (1 -> dim)   3×3 conv
        |
    [Stage 1: CNN Denoising]
    ResidualBlocks × encoder_blocks  (local feature extraction)
        |
    [Stage 2: Global Denoising — Restormer]
    RestormerBlocks × restormer_blocks  (global channel attention)
    -> Captures long-range speckle correlations
        |
    [Stage 3: SR Feature Extraction — Swin]
    SwinBlockPairs × (swin_blocks // 2)   (local spatial attention)
    -> Enhances local edge and texture features for upsampling
        |
    [Stage 4: Feature Fusion]
    1×1 Conv + global residual add
        |
    [Stage 5: 2× Upsampling]
    PixelShuffleUpsample
        |
    [Stage 6: Reconstruction]
    Conv → ReLU → Conv → Tanh
        |
    Output: (B, 1, 2H, 2W)   Restored GT in [-1, +1]

Information Flow
-----------------
Two global residual connections are used:
  1. Shallow feature skip: added after Stage 3, before upsampling
     -> Ensures low-level features reach the output regardless of depth
  2. The stages build progressively: CNN → Restormer → Swin → Upsample
     -> Avoids feature collapse in deep networks

Complexity Analysis
-------------------
For typical training patches (LR: 64×64, dim=64):
  CNN ResBlocks:    ~O(n_blocks × C² × HW)    local, fast
  Restormer MDTA:   ~O(n_blocks × C² × HW)    global channels, same cost!
  Swin W-MSA:       ~O(n_pairs × HW × C)      local spatial, linear in HW

Total parameters (dim=64, 4 ResBlocks, 4 Restormer, 2 Swin pairs): ~3.5M

VRAM budget for RTX 2050 (4.3 GB):
  - Batch=16, patch=64×64, AMP: ~1.5 GB for forward + 2.5 GB for gradients
  - Safely within 4.3 GB budget
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from models.baseline_cnn   import ResidualBlock, PixelShuffleUpsample
from models.restormer_block import RestormerBlock
from models.swin_block      import SwinBlockPair


class AMSRNet(nn.Module):
    """
    Adaptive Multi-Expert Semiconductor Restoration Network.

    Input  : (B, in_channels, H,   W  )  NoisyLR in ~[-1.4, +1.4]
    Output : (B, in_channels, H*2, W*2)  Restored GT in [-1, +1]

    Parameters
    ----------
    in_channels      : int    Image channels (1 for grayscale).
    dim              : int    Internal feature dimension.
    encoder_blocks   : int    Number of CNN residual blocks (Stage 1).
    restormer_blocks : int    Number of Restormer blocks (Stage 2).
    swin_blocks      : int    Number of Swin block pairs (Stage 3).
                              Must be >= 1. Each pair = W-MSA + SW-MSA.
    num_heads        : int    Attention heads for both Restormer and Swin.
    window_size      : int    Swin window size (must divide LR patch size).
    ffn_expansion    : float  Restormer FFN expansion ratio.
    mlp_ratio        : float  Swin MLP expansion ratio.
    scale            : int    Super-resolution upscaling factor.
    res_scale        : float  CNN residual block scaling.
    """

    def __init__(
        self,
        in_channels:      int   = cfg.CHANNELS,
        dim:              int   = cfg.AMSRNET_DIM,
        encoder_blocks:   int   = cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks: int   = cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks:      int   = cfg.AMSRNET_SWIN_BLOCKS,
        num_heads:        int   = cfg.AMSRNET_NUM_HEADS,
        window_size:      int   = cfg.AMSRNET_WINDOW_SIZE,
        ffn_expansion:    float = 2.66,
        mlp_ratio:        float = 2.0,
        scale:            int   = cfg.SCALE,
        res_scale:        float = 0.1,
    ) -> None:
        super().__init__()
        self.scale = scale

        # ------------------------------------------------------------------
        # Stage 0: Shallow feature extraction
        # ------------------------------------------------------------------
        self.shallow = nn.Conv2d(in_channels, dim, 3, padding=1, bias=True)

        # ------------------------------------------------------------------
        # Stage 1: CNN encoder (local feature extraction + initial denoising)
        # Re-uses the validated ResidualBlock from Phase 3.
        # ------------------------------------------------------------------
        self.cnn_encoder = nn.Sequential(
            *[ResidualBlock(dim, reduction=8, res_scale=res_scale)
              for _ in range(encoder_blocks)]
        )

        # ------------------------------------------------------------------
        # Stage 2: Restormer blocks (global channel attention for denoising)
        # Operates at the same LR resolution -- no downsampling.
        # ------------------------------------------------------------------
        self.restormer = nn.Sequential(
            *[RestormerBlock(
                dim           = dim,
                num_heads     = num_heads,
                ffn_expansion = ffn_expansion,
                bias          = cfg.AMSRNET_NUM_HEADS > 0,  # use bias if heads > 0
            ) for _ in range(restormer_blocks)]
        )

        # ------------------------------------------------------------------
        # Stage 3: Swin block pairs (local spatial attention for SR prep)
        # ------------------------------------------------------------------
        self.swin_transformer = nn.Sequential(
            *[SwinBlockPair(
                dim         = dim,
                num_heads   = num_heads,
                window_size = window_size,
                mlp_ratio   = mlp_ratio,
                drop        = 0.0,
            ) for _ in range(max(1, swin_blocks // 2))]
        )

        # ------------------------------------------------------------------
        # Stage 4: Feature fusion conv (combines all stage outputs)
        # ------------------------------------------------------------------
        self.fusion = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

        # ------------------------------------------------------------------
        # Stage 5: 2× upsampling (PixelShuffle, no checkerboard artifacts)
        # Re-uses the validated PixelShuffleUpsample from Phase 3.
        # ------------------------------------------------------------------
        self.upsample = PixelShuffleUpsample(dim, scale=scale)

        # ------------------------------------------------------------------
        # Stage 6: Reconstruction head
        # ------------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, in_channels, 3, padding=1, bias=True),
        )

        # ------------------------------------------------------------------
        # Weight initialisation (Kaiming for all Conv/Linear layers)
        # ------------------------------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for Conv2d, with Zero-Init for the final layer."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize the final reconstruction layer 
        # so the network starts as an exact identity mapping (preventing dead residual plateau)
        if isinstance(self.head[-1], nn.Conv2d):
            nn.init.zeros_(self.head[-1].weight)
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all stages.

        Parameters
        ----------
        x : torch.Tensor  (B, 1, H, W)  NoisyLR in [-1.4, +1.4]

        Returns
        -------
        torch.Tensor  (B, 1, H*2, W*2)  Restored GT in [-1, +1]
        """
        # Stage 0: shallow features
        shallow = self.shallow(x)             # (B, dim, H, W)

        # Stage 1: CNN local denoising
        feat = self.cnn_encoder(shallow)      # (B, dim, H, W)

        # Stage 2: Restormer global denoising
        feat = self.restormer(feat)           # (B, dim, H, W)

        # Stage 3: Swin spatial attention (prepares for SR)
        feat = self.swin_transformer(feat)    # (B, dim, H, W)

        # Stage 4: Fusion + global residual
        feat = self.fusion(feat) + shallow    # (B, dim, H, W)

        # Stage 5: 2x upsampling
        feat = self.upsample(feat)            # (B, dim, H*2, W*2)

        # Stage 6: Reconstruction
        out = self.head(feat)                # (B, 1, H*2, W*2)
        
        # Global image-space residual connection
        # The network only needs to learn the high-frequency residual (sharpening + noise removal).
        # This guarantees a strong baseline PSNR and prevents collapse to a gray image.
        base = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return out + base

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        n = self.count_parameters()
        n_cnn       = sum(p.numel() for m in self.cnn_encoder.children()
                          for p in m.parameters())
        n_restormer = sum(p.numel() for m in self.restormer.children()
                          for p in m.parameters())
        n_swin      = sum(p.numel() for m in self.swin_transformer.children()
                          for p in m.parameters())
        return (
            f"AMSRNet(\n"
            f"  dim              = {self.shallow.out_channels}\n"
            f"  CNN ResBlocks    = {len(list(self.cnn_encoder.children()))} "
            f"({n_cnn/1e6:.3f}M)\n"
            f"  Restormer Blocks = {len(list(self.restormer.children()))} "
            f"({n_restormer/1e6:.3f}M)\n"
            f"  Swin Pairs       = {len(list(self.swin_transformer.children()))} "
            f"({n_swin/1e6:.3f}M)\n"
            f"  scale            = {self.scale}x\n"
            f"  TOTAL params     = {n:,}  ({n/1e6:.2f}M)\n"
            f")"
        )
