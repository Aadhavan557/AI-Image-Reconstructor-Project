"""
models/swin_block.py
=====================
AMSR-Net - Phase 7: Swin Transformer Building Blocks
------------------------------------------------------
Implements Window-based Multi-Head Self-Attention (W-MSA) and
Shifted-Window Multi-Head Self-Attention (SW-MSA) from Swin Transformer
(Liu et al., ICCV 2021), adapted for image restoration / SR.

Why Swin for AMSR-Net?
-----------------------
While Restormer handles GLOBAL context efficiently via transposed attention,
it cannot model FINE LOCAL SPATIAL RELATIONSHIPS because its attention
operates over channels, not spatial positions.

For super-resolution, local spatial structure is critical:
    - The position of a circuit line edge within a 16×16 window
      determines whether the upsampled output has a sharp or blurry edge.
    - Global attention cannot preserve sub-pixel accuracy; local attention can.

Swin Transformer solves this by restricting self-attention to non-overlapping
local WINDOWS of size window_size × window_size. This gives:
    Complexity: O(window_size² × HW × C / window_size²) = O(HW × C)
    which scales linearly with image size -- efficient and local.

The SHIFTED window alternation ensures cross-window information exchange:
    - Even layers:  standard window partition (W-MSA)
    - Odd  layers:  shifted window partition  (SW-MSA)

Together, they model:
    Restormer: global channel dependencies  (denoising)
    Swin:      local spatial dependencies   (SR, edge sharpening)

Architecture of each Swin Block
---------------------------------

    Input x (B, H, W, C)  [note: HWC format for this module]
        |
    LayerNorm
        |
    W-MSA or SW-MSA
    (+ relative position bias)
        |
    + x (residual)
        |
    LayerNorm
        |
    MLP (2-layer FFN with GELU)
        |
    + (residual)
        |
    Output (B, H, W, C)

Relative Position Bias
-----------------------
Swin adds a learnable relative position bias B to the attention logits:
    Attention = softmax( QK^T / scale + B )
This allows the model to encode the relative spatial offset between
query and key tokens, which is critical for SR because the output
pixel positions depend on their spatial relationship to input tokens.

Implementation Notes
--------------------
- We use the BHWC (batch, height, width, channels) format internally,
  converting from BCHW at the block boundary.
- Cyclic shift for SW-MSA is implemented via torch.roll.
- Masking for shifted windows prevents attention across window boundaries.
- We implement relative position bias as a learned lookup table.

References
----------
- Liu et al. (2021): "Swin Transformer: Hierarchical Vision Transformer
  using Shifted Windows." ICCV 2021. https://arxiv.org/abs/2103.14030
- Liang et al. (2021): "SwinIR: Image Restoration Using Swin Transformer."
  ICCVW 2021. -- adaptation for image restoration tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ===========================================================================
# Window partition / reverse utilities
# ===========================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition feature map into non-overlapping windows.

    Parameters
    ----------
    x           : torch.Tensor  (B, H, W, C)
    window_size : int           Window side length.

    Returns
    -------
    torch.Tensor  (num_windows * B, window_size, window_size, C)

    The total number of windows is (H // window_size) * (W // window_size).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Permute: (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows:     torch.Tensor,
    window_size: int,
    H:           int,
    W:           int,
) -> torch.Tensor:
    """
    Reverse window partition back to the full feature map.

    Parameters
    ----------
    windows     : torch.Tensor  (num_windows * B, window_size, window_size, C)
    window_size : int
    H, W        : int  Original spatial dimensions.

    Returns
    -------
    torch.Tensor  (B, H, W, C)
    """
    B_times_nW = windows.shape[0]
    nH = H // window_size
    nW = W // window_size
    B  = B_times_nW // (nH * nW)

    x = windows.view(B, nH, nW, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


# ===========================================================================
# Window Multi-Head Self-Attention (W-MSA / SW-MSA)
# ===========================================================================

class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention with relative position bias.

    Handles both W-MSA (shift=0) and SW-MSA (shift=window_size//2).

    Parameters
    ----------
    dim         : int   Feature channel count (must be divisible by num_heads).
    window_size : int   Spatial window size (assumes square windows).
    num_heads   : int   Number of attention heads.
    qkv_bias    : bool  Add bias to QKV projections.
    attn_drop   : float Dropout on attention weights.
    proj_drop   : float Dropout on output projection.
    """

    def __init__(
        self,
        dim:         int   = 64,
        window_size: int   = 8,
        num_heads:   int   = 4,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = head_dim ** -0.5   # 1 / sqrt(d_k)

        # QKV projection
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # ------------------------------------------------------------------
        # Relative position bias table
        # ------------------------------------------------------------------
        # For a window of size (ws, ws), the relative position range is
        # [-(ws-1), +(ws-1)] in both H and W, giving (2*ws-1)² distinct values.
        ws = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Pre-compute relative position indices
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flat = torch.flatten(coords, 1)    # (2, ws*ws)

        # Relative coords: (2, ws*ws, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()   # (ws*ws, ws*ws, 2)

        # Shift to start from 0
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        # Convert 2D relative position to 1D index
        rel[:, :, 0] *= 2 * ws - 1
        rel_idx = rel.sum(-1)   # (ws*ws, ws*ws)
        self.register_buffer("relative_position_index", rel_idx)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : torch.Tensor  (num_windows * B, window_size², C)
        mask : torch.Tensor or None  Attention mask for shifted windows.

        Returns
        -------
        torch.Tensor  (num_windows * B, window_size², C)
        """
        nB, N, C = x.shape   # nB = num_windows * B
        heads    = self.num_heads
        head_dim = C // heads

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(nB, N, 3, heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, nB, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (nB, heads, N, N)

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)                         # (N, N, heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = attn + bias

        # Apply mask for shifted windows (SW-MSA)
        if mask is not None:
            nW = mask.shape[0]
            B_ = nB // nW
            attn = attn.view(B_, nW, heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(nB, heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values
        x = (attn @ v).transpose(1, 2).reshape(nB, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ===========================================================================
# Swin Transformer Block (W-MSA or SW-MSA)
# ===========================================================================

class SwinBlock(nn.Module):
    """
    Single Swin Transformer block.

    Alternates between W-MSA (shift_size=0) and SW-MSA (shift_size=ws//2)
    depending on the `shift` flag.

    Input/output are in BCHW format for seamless integration with the CNN
    backbone. The BHWC conversion happens internally.

    Parameters
    ----------
    dim         : int    Feature channels.
    num_heads   : int    Number of attention heads.
    window_size : int    Spatial window size.
    shift       : bool   If True, use shifted window (SW-MSA).
    mlp_ratio   : float  MLP expansion ratio.
    drop        : float  Dropout probability.
    """

    def __init__(
        self,
        dim:         int   = 64,
        num_heads:   int   = 4,
        window_size: int   = 8,
        shift:       bool  = False,
        mlp_ratio:   float = 2.0,
        drop:        float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.shift_size  = window_size // 2 if shift else 0
        self.shift       = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim         = dim,
            window_size = window_size,
            num_heads   = num_heads,
            attn_drop   = drop,
            proj_drop   = drop,
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Attention mask is pre-computed based on spatial size -- see _build_mask
        self._mask_H: int = 0
        self._mask_W: int = 0
        self.register_buffer("attn_mask", None, persistent=False)

    def _build_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Build the attention mask for SW-MSA.

        The mask ensures that attention does NOT flow between pixels from
        different "real" windows (i.e., windows that were split by the cyclic shift).

        Without this mask, pixels at the right edge of one real window
        could attend to pixels at the left edge of the next -- which would
        create incorrect cross-window information mixing.

        Returns None for W-MSA (no shift, no mask needed).
        """
        if self.shift_size == 0:
            return None   # W-MSA: no masking needed

        ws   = self.window_size
        ss   = self.shift_size

        # Create a label map: each region gets a different integer
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        label = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = label
                label += 1

        # Partition the label map into windows
        mask_windows = window_partition(img_mask, ws)      # (nW, ws, ws, 1)
        mask_windows = mask_windows.view(-1, ws * ws)       # (nW, ws²)

        # Compute pairwise difference: 0 = same region, !=0 = different region
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # Apply large negative value to mask out cross-region attention
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask   # (nW, ws², ws²)

    def _maybe_pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Pad H and W to be divisible by window_size if needed.
        Returns the padded tensor and the padding values.
        """
        _, H, W, _ = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad H and W
        return x, H, W, pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  (B, C, H, W)

        Returns
        -------
        torch.Tensor  (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Convert to BHWC for Swin operations
        x_hwc = x.permute(0, 2, 3, 1)   # (B, H, W, C)

        # Pad to window_size multiple if necessary
        x_pad, orig_H, orig_W, pad_h, pad_w = self._maybe_pad(x_hwc)
        _, Hp, Wp, _ = x_pad.shape

        # Build mask if needed (cache by H,W)
        if self.shift_size > 0 and (Hp != self._mask_H or Wp != self._mask_W):
            self.attn_mask = self._build_mask(Hp, Wp, x.device)
            self._mask_H   = Hp
            self._mask_W   = Wp

        # ---- Self-attention branch ----
        shortcut = x_pad   # (B, Hp, Wp, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted = torch.roll(x_pad, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_pad

        # Partition into windows: (nW*B, ws, ws, C)
        ws = self.window_size
        windows = window_partition(shifted, ws)
        windows = windows.view(-1, ws * ws, C)   # (nW*B, ws², C)

        # Apply window attention
        attn_out = self.attn(self.norm1(windows), mask=self.attn_mask)

        # Reverse windows back to feature map
        attn_out = attn_out.view(-1, ws, ws, C)
        shifted_out = window_reverse(attn_out, ws, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_attn = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted_out

        # Residual + MLP
        x_out = shortcut + x_attn
        x_out = x_out + self.mlp(self.norm2(x_out))

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x_out = x_out[:, :orig_H, :orig_W, :].contiguous()

        # Convert back to BCHW
        return x_out.permute(0, 3, 1, 2)


# ===========================================================================
# Swin Block Pair (W-MSA + SW-MSA)
# ===========================================================================

class SwinBlockPair(nn.Module):
    """
    Standard Swin Transformer block pair: W-MSA followed by SW-MSA.

    The pair ensures full cross-window coverage in 2 steps:
      Step 1 (W-MSA):  each window attends within itself
      Step 2 (SW-MSA): shifted windows allow inter-window information flow

    Parameters
    ----------
    dim         : int    Feature channels.
    num_heads   : int    Number of attention heads.
    window_size : int    Window size.
    mlp_ratio   : float  MLP expansion ratio.
    drop        : float  Dropout rate.
    """

    def __init__(
        self,
        dim:         int   = 64,
        num_heads:   int   = 4,
        window_size: int   = 8,
        mlp_ratio:   float = 2.0,
        drop:        float = 0.0,
    ) -> None:
        super().__init__()
        self.w_msa  = SwinBlock(dim, num_heads, window_size, shift=False,
                                mlp_ratio=mlp_ratio, drop=drop)
        self.sw_msa = SwinBlock(dim, num_heads, window_size, shift=True,
                                mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_msa(x)
        x = self.sw_msa(x)
        return x
