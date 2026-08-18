"""
models/restormer_block.py
==========================
AMSR-Net - Phase 7: Restormer Building Blocks
----------------------------------------------
Implements the two core sub-modules from Restormer (Zamir et al., CVPR 2022):
  1. MDTA  -- Multi-DConv Head Transposed Attention
  2. GDFN  -- Gated-DConv Feed-Forward Network

Why Restormer for AMSR-Net?
----------------------------
Standard Transformer self-attention has complexity O(H²W²) because it
computes pairwise similarity across ALL spatial positions. For a 128×128
feature map this is 268 million attention pairs -- completely intractable.

Restormer's key insight: transpose the attention -- compute attention
across CHANNELS instead of spatial positions:
    Complexity: O(C² * HW)  not  O(H²W² * C)

For AMSR-Net:
    C = 64  channels    ->  64² * 128 * 128  = 67M  ops  (tractable)
    H = 128, W = 128    ->  128²* 128²* 64   = 1.7B ops  (intractable)

So Restormer blocks model GLOBAL dependencies efficiently, which is
exactly what is needed for the DENOISING stage: speckle noise is
spatially correlated across large regions, so the model needs to
"see" the whole image context to identify and remove it.

Architecture of each Restormer Block
--------------------------------------

    Input x (B, C, H, W)
        |
    LayerNorm
        |
    MDTA (Multi-DConv Head Transposed Attention)
        |
    + x (residual)
        |
    LayerNorm
        |
    GDFN (Gated-DConv Feed-Forward Network)
        |
    + (residual)
        |
    Output (B, C, H, W)

MDTA Details
-------------
1. Project x -> Q, K, V  via 1×1 conv (parameter efficient)
2. Apply 3×3 depthwise conv to each (captures local spatial context)
3. Reshape to (B, heads, C//heads, HW) -- "transposed" form
4. Attention: A = softmax( Q^T K / scale )  in channel space
   A has shape (B, heads, C//heads, C//heads) -- small!
5. Output = A * V, reshape back to (B, C, H, W)

The key: attention is over the C//heads dimension, not HW.
This means each "token" is a spatial position, attended to
across all other positions via the channel axis.

GDFN Details
-------------
A gated version of the standard FFN:
    output = gate( W1 * x ) * ( W2 * x )

where gate() is GELU and W1, W2 are depthwise conv layers.
The gating mechanism controls information flow and acts as
a learned attention over features -- analogous to LSTM gates.

References
----------
- Zamir et al. (2022): "Restormer: Efficient Transformer for High-Resolution
  Image Restoration." CVPR 2022. https://arxiv.org/abs/2111.09881
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Layer Norm for (B, C, H, W) tensors
# ===========================================================================

class LayerNorm2d(nn.Module):
    """
    Layer normalisation that operates on (B, C, H, W) tensors.

    Standard nn.LayerNorm expects (B, *, C) format. This wrapper
    permutes to (B, H, W, C), applies LayerNorm over C, then permutes back.

    Why LayerNorm over BatchNorm for restoration?
    ----------------------------------------------
    BatchNorm normalises over the batch dimension, which introduces a
    dependency on batch size and creates train/test discrepancy.
    LayerNorm normalises over the channel dimension independently for
    each sample -- appropriate for image restoration where each image
    has its own noise characteristics (the batch statistics should NOT
    be shared).
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ===========================================================================
# MDTA: Multi-DConv Head Transposed Attention
# ===========================================================================

class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention (Zamir et al., 2022).

    Computes attention over the CHANNEL axis, not the spatial axis.
    This gives O(C²HW) complexity instead of O(H²W²C).

    Parameters
    ----------
    dim        : int  Feature channel count.
    num_heads  : int  Number of attention heads.
    bias       : bool Whether to use bias in convolutions.
    """

    def __init__(
        self,
        dim:       int  = 64,
        num_heads: int  = 4,
        bias:      bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Learnable per-head temperature (scaling factor for attention scores)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1×1 conv: project x to Q, K, V simultaneously (3x channels out)
        self.qkv      = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        # 3×3 depthwise conv: captures local spatial context in Q, K, V
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3,
                                     kernel_size=3, stride=1, padding=1,
                                     groups=dim * 3, bias=bias)
        # Output projection
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, C, H, W)

        Returns
        -------
        torch.Tensor  (B, C, H, W) attention output
        """
        B, C, H, W = x.shape

        # Project + depthwise: (B, 3C, H, W)
        qkv  = self.qkv_dwconv(self.qkv(x))
        # Split into Q, K, V: each (B, C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to multi-head transposed form: (B, heads, C//heads, HW)
        # Each "token" in attention is a channel-slice of shape (C//heads,)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        # L2-normalise Q and K for stable dot products
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Transposed attention: (B, heads, C//heads, C//heads)
        # This is attention between CHANNELS (not spatial positions)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # Apply attention to V: (B, heads, C//heads, HW)
        out = attn @ v

        # Reshape back to (B, C, H, W)
        out = out.reshape(B, C, H, W)
        return self.project_out(out)


# ===========================================================================
# GDFN: Gated-DConv Feed-Forward Network
# ===========================================================================

class GDFN(nn.Module):
    """
    Gated-DConv Feed-Forward Network (Zamir et al., 2022).

    Structure
    ---------
        x -> 1×1 Conv (expand to ffn_dim*2) -> 3×3 DWConv -> split into gate + feat
        gate = GELU(gate_half)
        output = gate * feat_half
        -> 1×1 Conv (project back to dim)

    Why gating?
    -----------
    The gating mechanism (sigmoid or GELU activation on one branch,
    multiplied with the other) acts as a learnable soft gate that
    selects which features to pass through. This:
    1. Improves gradient flow (similar to highway networks / LSTM)
    2. Reduces the effective rank of the FFN output (prevents over-fitting)
    3. Acts as a learned attention over feature activations

    Parameters
    ----------
    dim            : int    Input/output channel count.
    ffn_expansion  : float  Expansion ratio for the intermediate FFN size.
    bias           : bool   Whether to use bias.
    """

    def __init__(
        self,
        dim:           int   = 64,
        ffn_expansion: float = 2.66,
        bias:          bool  = False,
    ) -> None:
        super().__init__()
        # Intermediate (hidden) dimension -- expand for capacity
        hidden = int(dim * ffn_expansion)

        # Project in: 1x1 conv, expand to 2*hidden (gate + feature)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        # Depthwise 3x3 on the full 2*hidden tensor
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2,
                                     kernel_size=3, stride=1, padding=1,
                                     groups=hidden * 2, bias=bias)
        # Project back to dim
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1    = self.project_in(x)       # (B, 2*hidden, H, W)
        x1    = self.dwconv(x1)          # local context
        x1, x2 = x1.chunk(2, dim=1)     # split: gate and feature
        x1    = F.gelu(x1) * x2         # gated activation
        return self.project_out(x1)


# ===========================================================================
# Restormer Block (MDTA + GDFN)
# ===========================================================================

class RestormerBlock(nn.Module):
    """
    Single Restormer transformer block.

    Structure
    ---------
        x -> LN -> MDTA -> + x  (attention residual)
          -> LN -> GDFN -> + x  (ffn residual)

    Parameters
    ----------
    dim           : int    Feature channels.
    num_heads     : int    Number of attention heads.
    ffn_expansion : float  FFN expansion ratio.
    bias          : bool   Use bias in convolutions.
    """

    def __init__(
        self,
        dim:           int   = 64,
        num_heads:     int   = 4,
        ffn_expansion: float = 2.66,
        bias:          bool  = False,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn  = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn   = GDFN(dim=dim, ffn_expansion=ffn_expansion, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # attention residual
        x = x + self.ffn(self.norm2(x))    # FFN residual
        return x
