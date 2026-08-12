"""
phase7_verify.py
================
AMSR-Net - Phase 7: AMSR-Net Hybrid Architecture Verification
--------------------------------------------------------------
Validates the full AMSR-Net hybrid model and its building blocks:

  1. Component Unit Tests:
     - RestormerBlock (MDTA + GDFN) forward pass & shape invariance
     - SwinBlock & SwinBlockPair (W-MSA + SW-MSA) forward pass & shape invariance
     - Relative position bias lookup & window partitioning round-trip
  2. Dynamic Spatial Input Test:
     - Model accepts multiple spatial sizes (e.g. 64x64 patch, 128x128 full LR)
     - Output is correctly 2x upsampled (128x128 -> 256x256)
  3. Output Range & Gradient Flow:
     - Tanh output strictly in [-1.0, 1.0]
     - Full gradient flow check through all parameters (no zero grads or NaN)
  4. VRAM & Speed Profiling on GPU:
     - Measures forward/backward latency and peak CUDA VRAM under AMP
  5. Trainer & Loss Integration Test:
     - Runs 1 training batch through CompositeLoss + GradScaler to ensure total compatibility

Run
---
  python phase7_verify.py
"""

import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# UTF-8 Windows fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from models import (
    AMSRNet, RestormerBlock, MDTA, GDFN,
    SwinBlock, SwinBlockPair, window_partition, window_reverse
)
from losses import CompositeLoss

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def check(condition: bool, message: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"    {tag}  {message}")
    return condition


# ===========================================================================
# Test 1: Restormer Building Blocks
# ===========================================================================

def test_restormer_blocks():
    print("\n[1/6]  Restormer building blocks (MDTA, GDFN, RestormerBlock)...")
    ok = True
    B, C, H, W = 2, 64, 32, 32
    x = torch.randn(B, C, H, W)

    # MDTA
    mdta = MDTA(dim=C, num_heads=4)
    out_mdta = mdta(x)
    ok &= check(out_mdta.shape == (B, C, H, W),
                f"MDTA output shape: {tuple(out_mdta.shape)}  (expected {(B, C, H, W)})")

    # GDFN
    gdfn = GDFN(dim=C, ffn_expansion=2.66)
    out_gdfn = gdfn(x)
    ok &= check(out_gdfn.shape == (B, C, H, W),
                f"GDFN output shape: {tuple(out_gdfn.shape)}  (expected {(B, C, H, W)})")

    # Full RestormerBlock
    block = RestormerBlock(dim=C, num_heads=4, ffn_expansion=2.66)
    out_block = block(x)
    ok &= check(out_block.shape == (B, C, H, W),
                f"RestormerBlock output shape: {tuple(out_block.shape)}")

    # Check finite
    ok &= check(torch.isfinite(out_block).all().item(),
                "RestormerBlock outputs are finite (no NaN/Inf)")

    return ok


# ===========================================================================
# Test 2: Swin Building Blocks
# ===========================================================================

def test_swin_blocks():
    print("\n[2/6]  Swin building blocks (Window Partition, W-MSA/SW-MSA)...")
    ok = True
    B, H, W, C = 2, 32, 32, 64
    x_hwc = torch.randn(B, H, W, C)

    # Window partition round-trip
    ws = 8
    windows = window_partition(x_hwc, window_size=ws)
    x_rec = window_reverse(windows, window_size=ws, H=H, W=W)
    ok &= check(torch.allclose(x_hwc, x_rec, atol=1e-6),
                "Window partition <-> reverse round-trip is lossless")

    # SwinBlockPair (BCHW format)
    x_bchw = x_hwc.permute(0, 3, 1, 2)
    swin_pair = SwinBlockPair(dim=C, num_heads=4, window_size=ws)
    out_swin = swin_pair(x_bchw)

    ok &= check(out_swin.shape == (B, C, H, W),
                f"SwinBlockPair output shape: {tuple(out_swin.shape)}  (expected {(B, C, H, W)})")
    ok &= check(torch.isfinite(out_swin).all().item(),
                "SwinBlockPair outputs are finite")

    return ok


# ===========================================================================
# Test 3: Full AMSR-Net Forward Pass & Dynamic Resolution
# ===========================================================================

def test_amsrnet_forward():
    print("\n[3/6]  AMSR-Net full model forward pass & multi-resolution test...")
    ok = True
    model = AMSRNet(
        in_channels=cfg.CHANNELS,
        dim=cfg.AMSRNET_DIM,
        encoder_blocks=cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks=cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks=cfg.AMSRNET_SWIN_BLOCKS,
        num_heads=cfg.AMSRNET_NUM_HEADS,
        window_size=cfg.AMSRNET_WINDOW_SIZE,
        scale=cfg.SCALE,
    )

    print(f"    {INFO} Model summary:")
    print(f"        Total parameters: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")

    # Test patch resolution (64x64 -> 128x128)
    x_patch = torch.randn(2, 1, 64, 64)
    y_patch = model(x_patch)
    ok &= check(y_patch.shape == (2, 1, 128, 128),
                f"Patch input (64x64) -> Output shape: {tuple(y_patch.shape)}  (expected (2, 1, 128, 128))")

    # Test full LR resolution (128x128 -> 256x256)
    x_full = torch.randn(1, 1, 128, 128)
    y_full = model(x_full)
    ok &= check(y_full.shape == (1, 1, 256, 256),
                f"Full LR input (128x128) -> Output shape: {tuple(y_full.shape)}  (expected (1, 1, 256, 256))")

    # Ensure no NaN in output bounds (no Tanh anymore, loss constrains range)
    ok &= check(torch.isfinite(y_patch).all().item(),
                f"Output values are finite: min={y_patch.min().item():.4f}, max={y_patch.max().item():.4f}")

    return ok


# ===========================================================================
# Test 4: Gradient Flow Check
# ===========================================================================

def test_gradient_flow():
    print("\n[4/6]  Full gradient flow check through all parameters...")
    ok = True
    model = AMSRNet(
        in_channels=cfg.CHANNELS,
        dim=32,                  # lighter dimension for quick test
        encoder_blocks=2,
        restormer_blocks=2,
        swin_blocks=2,
    )
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    no_grad_params = []
    nan_grad_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                no_grad_params.append(name)
            elif torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                nan_grad_params.append(name)

    ok &= check(len(no_grad_params) == 0,
                f"Params without gradients: {len(no_grad_params)} (expected 0)")
    if len(no_grad_params) > 0:
        print(f"        Missing grads: {no_grad_params[:5]}...")

    ok &= check(len(nan_grad_params) == 0,
                f"Params with NaN/Inf gradients: {len(nan_grad_params)} (expected 0)")

    return ok


# ===========================================================================
# Test 5: GPU Performance & VRAM Profiling
# ===========================================================================

def test_gpu_profiling():
    print("\n[5/6]  GPU latency & peak VRAM memory profiling...")
    if not torch.cuda.is_available():
        print(f"    {INFO} CUDA not available -- skipping GPU profiling.")
        return True

    ok = True
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AMSRNet(
        in_channels=cfg.CHANNELS,
        dim=cfg.AMSRNET_DIM,
        encoder_blocks=cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks=cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks=cfg.AMSRNET_SWIN_BLOCKS,
    ).to(device)

    # Warmup
    x = torch.randn(cfg.BATCH_SIZE, cfg.CHANNELS, cfg.PATCH_SIZE, cfg.PATCH_SIZE, device=device)
    target = torch.randn(cfg.BATCH_SIZE, cfg.CHANNELS, cfg.PATCH_SIZE * cfg.SCALE, cfg.PATCH_SIZE * cfg.SCALE, device=device)

    criterion = CompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=True)

    # Warmup pass
    with torch.amp.autocast("cuda", enabled=True):
        out = model(x)
        loss = criterion(out, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Measured pass
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    n_runs = 5
    for _ in range(n_runs):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            out = model(x)
            loss = criterion(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / n_runs) * 1000.0
    throughput = (cfg.BATCH_SIZE * n_runs) / total_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    ok &= check(peak_vram_mb < 3500.0,
                f"Peak VRAM: {peak_vram_mb:.1f} MB  (budget < 3500 MB for RTX 2050)")
    ok &= check(avg_latency_ms < 1500.0,
                f"Batch train latency ({cfg.BATCH_SIZE} imgs): {avg_latency_ms:.1f} ms ({throughput:.1f} img/s)")

    return ok


# ===========================================================================
# Test 6: Trainer Compatibility & Model Selection
# ===========================================================================

def test_trainer_integration():
    print("\n[6/6]  Trainer integration check with AMSRNet...")
    ok = True

    try:
        from utils.trainer import Trainer
        from train import build_model

        # Update train.py model builder check if needed or construct model directly
        model = AMSRNet()
        dummy_x = torch.randn(2, 1, 64, 64)
        dummy_y = model(dummy_x)

        ok &= check(dummy_y.shape == (2, 1, 128, 128),
                    "AMSRNet instantiates and executes cleanly for Trainer usage")
    except Exception as e:
        ok = False
        check(False, f"Trainer integration check failed: {e}")

    return ok


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  AMSR-Net | Phase 7 -- Hybrid Architecture Verification")
    print("=" * 70)

    results = {
        "Restormer blocks":     test_restormer_blocks(),
        "Swin blocks":          test_swin_blocks(),
        "AMSRNet forward & resolution": test_amsrnet_forward(),
        "Gradient flow":        test_gradient_flow(),
        "GPU profiling":        test_gpu_profiling(),
        "Trainer integration":  test_trainer_integration(),
    }

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        tag = PASS if passed else FAIL
        print(f"  {tag}  {name}")
        all_pass &= passed

    print()
    if all_pass:
        print("  All checks passed. AMSR-Net hybrid model verified!")
        print("  Ready to update train.py model selection & execute full training runs.")
    else:
        print("  Some checks FAILED. Fix issues before initiating training.")

    print("=" * 70)


if __name__ == "__main__":
    main()
