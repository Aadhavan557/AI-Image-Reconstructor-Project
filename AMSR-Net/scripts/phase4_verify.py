"""
phase4_verify.py
================
AMSR-Net - Phase 4: Loss Functions Verification
-------------------------------------------------
Validates all loss function implementations:

  1. CharbonnierLoss -- value at zero, L1/L2 behaviour, gradient.
  2. SSIMLoss        -- perfect input gives 0 loss, shifted gives > 0.
  3. EdgeLoss        -- blurry prediction gives higher loss than sharp.
  4. CompositeLoss   -- correct weighted sum, backward pass.
  5. Loss scale      -- all losses are order-compatible for training.
  6. GPU transfer    -- all losses work on CUDA tensors.
  7. Visualisation   -- loss surface plots saved to outputs/.

Run
---
  python phase4_verify.py
"""

import sys
import os
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

# UTF-8 Windows fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from losses import CharbonnierLoss, SSIMLoss, EdgeLoss, CompositeLoss

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
# Test 1: CharbonnierLoss
# ===========================================================================

def test_charbonnier():
    print("\n[1/7]  CharbonnierLoss...")
    loss_fn = CharbonnierLoss(eps=1e-3)
    ok = True

    # -- At zero error, loss should be eps (not 0, because sqrt(0 + eps^2) = eps)
    x = torch.zeros(1, 1, 32, 32)
    l = loss_fn(x, x)
    ok &= check(abs(l.item() - 1e-3) < 1e-5,
                f"Loss at zero error = {l.item():.6f}  (expected eps=1e-3)")

    # -- Loss should be > 0 for non-zero error
    y = torch.ones(1, 1, 32, 32) * 0.5
    z = torch.zeros(1, 1, 32, 32)
    l_nonzero = loss_fn(y, z)
    ok &= check(l_nonzero.item() > 0.0,
                f"Loss at delta=0.5 = {l_nonzero.item():.6f}  (expected > 0)")

    # -- At delta=0.5, Charbonnier behaves L1-like (linear ~= delta),
    #    while MSE is quadratic (= delta^2 = 0.25).
    #    For delta in (0,1): L1 > L2 (Charbonnier > MSE is CORRECT).
    #    This is the L1 robustness property: less penalty than L2 for
    #    large errors relative to its own scale, but larger than MSE
    #    for the same absolute delta because L1 doesn't square it.
    delta   = torch.full((1, 1, 32, 32), 0.5)
    zero    = torch.zeros_like(delta)
    l_charb = loss_fn(delta, zero).item()
    l_mse   = nn.functional.mse_loss(delta, zero).item()
    # Charbonnier ~= 0.500 (L1-like), MSE = 0.250 (L2), so charb > mse here.
    # The key property: Charbonnier gradient is ~constant for large errors
    # (doesn't grow quadratically like MSE), making it robust to outliers.
    ok &= check(
        abs(l_charb - 0.5) < 0.01,
        f"Charbonnier(delta=0.5) = {l_charb:.4f}  ~= 0.5  (L1-like)"
    )
    ok &= check(
        l_charb > l_mse,
        f"Charbonnier={l_charb:.4f} > MSE={l_mse:.4f}  (linear, not quadratic)"
    )

    # -- Symmetric: loss(a,b) == loss(b,a)
    a = torch.randn(2, 1, 64, 64)
    b = torch.randn(2, 1, 64, 64)
    ok &= check(
        abs(loss_fn(a, b).item() - loss_fn(b, a).item()) < 1e-5,
        "Symmetric: L(a,b) == L(b,a)"
    )

    # -- Gradient flows
    p = torch.randn(2, 1, 64, 64, requires_grad=True)
    t = torch.randn(2, 1, 64, 64)
    loss_fn(p, t).backward()
    ok &= check(p.grad is not None and p.grad.abs().max() > 0,
                "Gradient flows through CharbonnierLoss")

    return ok


# ===========================================================================
# Test 2: SSIMLoss
# ===========================================================================

def test_ssim():
    print("\n[2/7]  SSIMLoss...")
    loss_fn = SSIMLoss(channels=1, data_range=2.0)
    ok = True

    # -- Perfect prediction: loss should be ~0
    x = torch.randn(1, 1, 128, 128)
    l_perfect = loss_fn(x, x).item()
    ok &= check(l_perfect < 0.01,
                f"Loss(x, x) = {l_perfect:.6f}  (expected ~0)")

    # -- Shifted image: loss > 0
    y = x + 0.3
    l_shifted = loss_fn(x, y).item()
    ok &= check(l_shifted > 0.01,
                f"Loss(x, x+0.3) = {l_shifted:.4f}  (expected > 0.01)")

    # -- Zero image vs ones: high loss
    zeros = torch.zeros(1, 1, 64, 64)
    ones  = torch.ones(1, 1,  64, 64)
    l_extreme = loss_fn(zeros, ones).item()
    ok &= check(l_extreme > 0.1,
                f"Loss(zeros, ones) = {l_extreme:.4f}  (expected high)")

    # -- ssim_score on perfect input = 1.0
    score = loss_fn.ssim_score(x, x).item()
    ok &= check(abs(score - 1.0) < 0.01,
                f"ssim_score(x, x) = {score:.6f}  (expected ~1.0)")

    # -- Gradient flows
    p = torch.randn(2, 1, 64, 64, requires_grad=True)
    t = torch.randn(2, 1, 64, 64)
    loss_fn(p, t).backward()
    ok &= check(p.grad is not None,
                "Gradient flows through SSIMLoss")

    return ok


# ===========================================================================
# Test 3: EdgeLoss
# ===========================================================================

def test_edge():
    print("\n[3/7]  EdgeLoss...")
    loss_fn = EdgeLoss(channels=1)
    ok = True

    # -- Sharp prediction vs GT should have lower edge loss than blurry
    gt = torch.zeros(1, 1, 64, 64)
    # Create a sharp vertical edge in the centre
    gt[:, :, :, 32:] = 1.0

    # Sharp prediction = gt (perfect)
    pred_sharp = gt.clone()
    l_sharp = loss_fn(pred_sharp, gt).item()

    # Blurry prediction = Gaussian-smoothed gt
    from torch.nn.functional import avg_pool2d
    pred_blurry = avg_pool2d(gt, kernel_size=5, stride=1, padding=2)
    l_blurry = loss_fn(pred_blurry, gt).item()

    ok &= check(l_sharp < l_blurry,
                f"Sharp loss ({l_sharp:.4f}) < Blurry loss ({l_blurry:.4f})")

    # -- Near-zero loss on identical inputs
    x = torch.randn(1, 1, 64, 64)
    l_same = loss_fn(x, x).item()
    ok &= check(l_same < 0.01,
                f"EdgeLoss(x, x) = {l_same:.6f}  (expected ~eps)")

    # -- Sobel buffers are on the correct device
    ok &= check(
        loss_fn.sobel_x.device.type == "cpu",
        f"Sobel kernels registered as buffer (device: {loss_fn.sobel_x.device})"
    )

    # -- Gradient flows
    p = torch.randn(2, 1, 64, 64, requires_grad=True)
    t = torch.randn(2, 1, 64, 64)
    loss_fn(p, t).backward()
    ok &= check(p.grad is not None,
                "Gradient flows through EdgeLoss")

    return ok


# ===========================================================================
# Test 4: CompositeLoss
# ===========================================================================

def test_composite():
    print("\n[4/7]  CompositeLoss...")
    loss_fn = CompositeLoss(
        w_charb=cfg.LOSS_CHARBONNIER_W,
        w_ssim=cfg.LOSS_SSIM_W,
        w_edge=cfg.LOSS_EDGE_W,
    )
    ok = True
    print(f"\n{loss_fn}")

    p = torch.randn(2, 1, 128, 128)
    t = torch.randn(2, 1, 128, 128)

    # -- forward() returns scalar
    total = loss_fn(p, t)
    ok &= check(total.ndim == 0,
                f"forward() returns scalar (shape={total.shape})")
    ok &= check(total.item() > 0,
                f"Total loss = {total.item():.6f}  (expected > 0)")

    # -- forward_detailed() returns correct keys
    detail = loss_fn.forward_detailed(p, t)
    for key in ["total", "charb", "ssim", "edge", "ssim_score"]:
        ok &= check(key in detail,
                    f"Key '{key}' in forward_detailed() output")

    # -- Manual weighted sum matches total
    manual = (
        cfg.LOSS_CHARBONNIER_W * detail["charb"]
        + cfg.LOSS_SSIM_W      * detail["ssim"]
        + cfg.LOSS_EDGE_W      * detail["edge"]
    )
    ok &= check(
        abs(manual.item() - detail["total"].item()) < 1e-4,
        f"Manual sum = {manual.item():.6f} ~= total = {detail['total'].item():.6f}"
    )

    # -- Backward pass
    p2 = torch.randn(2, 1, 128, 128, requires_grad=True)
    loss_fn(p2, t).backward()
    ok &= check(p2.grad is not None and p2.grad.abs().max() > 0,
                "Backward pass OK through CompositeLoss")

    return ok


# ===========================================================================
# Test 5: Loss scale compatibility
# ===========================================================================

def test_loss_scales():
    print("\n[5/7]  Loss scale compatibility...")

    charb  = CharbonnierLoss()
    ssim   = SSIMLoss(channels=1, data_range=2.0)
    edge   = EdgeLoss(channels=1)

    # Typical training scenario: random noise prediction vs clean GT
    pred   = torch.randn(4, 1, 128, 128) * 0.2
    target = torch.randn(4, 1, 128, 128) * 0.5

    l_c = charb(pred, target).item()
    l_s = ssim(pred, target).item()
    l_e = edge(pred, target).item()

    print(f"    {INFO}  Charbonnier  = {l_c:.4f}  (weight={cfg.LOSS_CHARBONNIER_W})")
    print(f"    {INFO}  SSIM         = {l_s:.4f}  (weight={cfg.LOSS_SSIM_W})")
    print(f"    {INFO}  Edge         = {l_e:.4f}  (weight={cfg.LOSS_EDGE_W})")

    weighted_c = cfg.LOSS_CHARBONNIER_W * l_c
    weighted_s = cfg.LOSS_SSIM_W        * l_s
    weighted_e = cfg.LOSS_EDGE_W        * l_e

    print(f"    {INFO}  Weighted Charbonnier = {weighted_c:.4f}")
    print(f"    {INFO}  Weighted SSIM        = {weighted_s:.4f}")
    print(f"    {INFO}  Weighted Edge        = {weighted_e:.4f}")

    # Check no term completely dominates (>100x the others)
    terms = [weighted_c, weighted_s, weighted_e]
    max_t, min_t = max(terms), min(terms)
    ratio = max_t / (min_t + 1e-8)

    ok = check(ratio < 100.0,
               f"Loss term ratio max/min = {ratio:.1f}x  (threshold: 100x)")
    return ok


# ===========================================================================
# Test 6: GPU transfer
# ===========================================================================

def test_gpu():
    print("\n[6/7]  GPU transfer...")

    if not torch.cuda.is_available():
        print(f"    {INFO}  CUDA not available -- skipping.")
        return True

    device = torch.device("cuda")
    ok = True

    losses = [
        ("CharbonnierLoss", CharbonnierLoss()),
        ("SSIMLoss",        SSIMLoss(channels=1, data_range=2.0)),
        ("EdgeLoss",        EdgeLoss(channels=1)),
        ("CompositeLoss",   CompositeLoss()),
    ]

    p = torch.randn(2, 1, 64, 64, device=device)
    t = torch.randn(2, 1, 64, 64, device=device)

    for name, loss_fn in losses:
        loss_fn = loss_fn.to(device)
        l = loss_fn(p, t)
        ok &= check(
            l.device.type == "cuda" and l.item() > 0,
            f"{name} on CUDA: loss={l.item():.4f}"
        )

    return ok


# ===========================================================================
# Test 7: Loss surface visualisation
# ===========================================================================

def test_visualisation():
    print("\n[7/7]  Generating loss surface visualisation...")

    # Create a synthetic GT image (sharp edge pattern)
    gt = torch.zeros(1, 1, 64, 64)
    gt[:, :, :, 32:] = 1.0                    # sharp vertical edge
    gt = gt * 2.0 - 1.0                        # map to [-1, +1]

    charb  = CharbonnierLoss()
    ssim_l = SSIMLoss(channels=1, data_range=2.0)
    edge_l = EdgeLoss(channels=1)

    # Sweep: add Gaussian noise at increasing levels
    noise_levels = np.linspace(0, 0.5, 30)
    l_charbs, l_ssims, l_edges = [], [], []

    for sigma in noise_levels:
        noisy = gt + torch.randn_like(gt) * sigma
        with torch.no_grad():
            l_charbs.append(charb(noisy, gt).item())
            l_ssims.append(ssim_l(noisy, gt).item())
            l_edges.append(edge_l(noisy, gt).item())

    # Sweep: add blur at increasing levels
    blur_levels = list(range(1, 15, 2))
    l_charbs_b, l_ssims_b, l_edges_b = [], [], []

    for ks in blur_levels:
        from torch.nn.functional import avg_pool2d
        blurry = avg_pool2d(gt, kernel_size=ks, stride=1, padding=ks // 2)
        with torch.no_grad():
            l_charbs_b.append(charb(blurry, gt).item())
            l_ssims_b.append(ssim_l(blurry, gt).item())
            l_edges_b.append(edge_l(blurry, gt).item())

    # Plot
    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3,
                            left=0.07, right=0.97, top=0.90, bottom=0.08)

    def plot_ax(ax, xs, ys, xlabel, ylabel, title, color):
        ax.set_facecolor("#161b22")
        ax.plot(xs, ys, color=color, linewidth=2)
        ax.set_xlabel(xlabel, color="#8b949e", fontsize=9)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
        ax.set_title(title, color="#e6edf3", fontsize=10, pad=5)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    plot_ax(fig.add_subplot(gs[0, 0]), noise_levels, l_charbs,
            "Noise sigma", "Loss", "Charbonnier vs Noise", "#58a6ff")
    plot_ax(fig.add_subplot(gs[0, 1]), noise_levels, l_ssims,
            "Noise sigma", "Loss", "SSIM Loss vs Noise", "#f78166")
    plot_ax(fig.add_subplot(gs[0, 2]), noise_levels, l_edges,
            "Noise sigma", "Loss", "Edge Loss vs Noise", "#3fb950")
    plot_ax(fig.add_subplot(gs[1, 0]), blur_levels, l_charbs_b,
            "Blur kernel size", "Loss", "Charbonnier vs Blur", "#58a6ff")
    plot_ax(fig.add_subplot(gs[1, 1]), blur_levels, l_ssims_b,
            "Blur kernel size", "Loss", "SSIM Loss vs Blur", "#f78166")
    plot_ax(fig.add_subplot(gs[1, 2]), blur_levels, l_edges_b,
            "Blur kernel size", "Loss", "Edge Loss vs Blur", "#3fb950")

    fig.suptitle(
        "AMSR-Net -- Phase 4: Loss Function Behaviour\n"
        "Row 1: vs. Noise level (sigma)  |  Row 2: vs. Blur level (kernel size)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=0.97
    )

    save_path = os.path.join(cfg.OUTPUTS_DIR, "phase4_loss_curves.png")
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [OK] Loss curve plots saved -> {save_path}")
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  AMSR-Net | Phase 4 -- Loss Functions Verification")
    print("=" * 70)

    results = {
        "CharbonnierLoss":     test_charbonnier(),
        "SSIMLoss":            test_ssim(),
        "EdgeLoss":            test_edge(),
        "CompositeLoss":       test_composite(),
        "Loss scale compat.":  test_loss_scales(),
        "GPU transfer":        test_gpu(),
        "Visualisation":       test_visualisation(),
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
        print("  All checks passed. Loss functions are verified.")
        print("  Ready to proceed to Phase 5: Metrics (PSNR + SSIM).")
    else:
        print("  Some checks FAILED. Fix errors above before Phase 5.")

    print("=" * 70)


if __name__ == "__main__":
    main()
