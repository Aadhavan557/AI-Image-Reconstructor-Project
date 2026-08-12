"""
phase5_verify.py
================
AMSR-Net - Phase 5: Metrics Verification
------------------------------------------
Validates PSNR, SSIM, and MetricTracker implementations:

  1. PSNRMetric -- known-value tests (identical, constant offset, noise).
  2. SSIMMetric -- perfect input, shifted, blurred, spatial map shape.
  3. PSNR vs TorchMetrics  -- cross-validate against torchmetrics if available.
  4. MetricTracker          -- accumulation correctness, best-epoch tracking.
  5. Denormalisation check  -- confirm metric pipeline ([-1,+1] -> [0,1]).
  6. GPU transfer           -- all metrics work on CUDA.
  7. Visualisation          -- SSIM map on a real image pair.

Run
---
  python phase5_verify.py
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

# UTF-8 Windows fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from metrics import PSNRMetric, SSIMMetric, MetricTracker
from dataset.semiconductor_dataset import (
    scan_directory, normalise_lr, normalise_gt, denormalise
)

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
# Test 1: PSNRMetric known values
# ===========================================================================

def test_psnr():
    print("\n[1/7]  PSNRMetric known-value tests...")
    psnr_fn = PSNRMetric(data_range=1.0)
    ok = True

    # -- Perfect prediction: PSNR should be very high (clamped to ~100 dB)
    x = torch.rand(1, 1, 128, 128)
    l = psnr_fn(x, x).item()
    ok &= check(l > 90.0,
                f"PSNR(x, x) = {l:.2f} dB  (expected > 90 dB)")

    # -- Known analytical value: constant offset of delta
    #    MSE = delta^2, PSNR = 10 * log10(1 / delta^2) = -20 * log10(delta)
    delta = 0.1
    zeros = torch.zeros(4, 1, 128, 128)
    const = torch.full((4, 1, 128, 128), delta)
    expected_psnr = -20.0 * math.log10(delta)   # = 20.0 dB
    measured_psnr = psnr_fn(zeros, const).item()
    ok &= check(
        abs(measured_psnr - expected_psnr) < 0.01,
        f"PSNR at delta=0.1: measured={measured_psnr:.4f} dB, "
        f"expected={expected_psnr:.4f} dB"
    )

    # -- Monotonicity: higher noise -> lower PSNR
    clean = torch.rand(2, 1, 64, 64)
    psnrs = []
    for sigma in [0.01, 0.05, 0.10, 0.20]:
        noisy = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)
        psnrs.append(psnr_fn(noisy, clean).item())
    ok &= check(
        all(psnrs[i] > psnrs[i+1] for i in range(len(psnrs)-1)),
        f"Monotone decreasing with noise: {[f'{p:.1f}' for p in psnrs]} dB"
    )

    # -- reduction='none' gives (B,) tensor
    psnr_none = PSNRMetric(data_range=1.0, reduction="none")
    a = torch.rand(5, 1, 64, 64)
    b = torch.rand(5, 1, 64, 64)
    scores = psnr_none(a, b)
    ok &= check(
        scores.shape == (5,),
        f"reduction='none' output shape = {tuple(scores.shape)}  (expected (5,))"
    )

    # -- static helper
    psnr_from_mse = PSNRMetric.mse_to_psnr(0.01, data_range=1.0)
    expected = 10.0 * math.log10(1.0 / 0.01)
    ok &= check(
        abs(psnr_from_mse - expected) < 0.001,
        f"mse_to_psnr(0.01) = {psnr_from_mse:.4f}  (expected {expected:.4f})"
    )

    return ok


# ===========================================================================
# Test 2: SSIMMetric
# ===========================================================================

def test_ssim():
    print("\n[2/7]  SSIMMetric known-value tests...")
    ssim_fn = SSIMMetric(channels=1, data_range=1.0)
    ok = True

    # -- Perfect match: SSIM = 1.0
    x = torch.rand(1, 1, 128, 128)
    score = ssim_fn(x, x).item()
    ok &= check(abs(score - 1.0) < 0.001,
                f"SSIM(x, x) = {score:.6f}  (expected ~1.0)")

    # -- Monotonicity: higher blur -> lower SSIM
    gt = torch.rand(1, 1, 64, 64)
    ssims = []
    for ks in [1, 3, 5, 7]:
        if ks == 1:
            blurry = gt
        else:
            blurry = torch.nn.functional.avg_pool2d(
                gt, kernel_size=ks, stride=1, padding=ks // 2
            )
        ssims.append(ssim_fn(blurry, gt).item())
    ok &= check(
        ssims[0] >= ssims[1] >= ssims[2],
        f"Monotone decreasing with blur: {[f'{s:.4f}' for s in ssims]}"
    )

    # -- Spatial map shape
    ssim_score, ssim_map = ssim_fn.forward_with_map(x, x)
    ok &= check(
        ssim_map.shape == (1, 1, 128, 128),
        f"SSIM map shape = {tuple(ssim_map.shape)}  (expected (1,1,128,128))"
    )
    ok &= check(
        ssim_score.item() > 0.99,
        f"SSIM map mean = {ssim_score.item():.6f}  (expected ~1.0)"
    )

    # -- reduction='none' gives (B,) tensor
    ssim_none = SSIMMetric(channels=1, data_range=1.0, reduction="none")
    a = torch.rand(5, 1, 64, 64)
    b = torch.rand(5, 1, 64, 64)
    scores = ssim_none(a, b)
    ok &= check(
        scores.shape == (5,),
        f"reduction='none' shape = {tuple(scores.shape)}  (expected (5,))"
    )
    ok &= check(
        scores.min().item() >= 0.0 and scores.max().item() <= 1.0,
        f"All SSIM values in [0,1]: min={scores.min():.4f} max={scores.max():.4f}"
    )

    return ok


# ===========================================================================
# Test 3: Cross-validation against TorchMetrics
# ===========================================================================

def test_crossvalidate():
    print("\n[3/7]  Cross-validation vs TorchMetrics (if available)...")

    try:
        from torchmetrics.functional import (
            peak_signal_noise_ratio as tm_psnr,
            structural_similarity_index_measure as tm_ssim,
        )
    except ImportError:
        print(f"    {INFO}  torchmetrics not installed -- skipping cross-validation.")
        print(f"    {INFO}  Install with: pip install torchmetrics")
        return True

    ok = True
    psnr_fn = PSNRMetric(data_range=1.0, reduction="mean")
    ssim_fn = SSIMMetric(channels=1, data_range=1.0, reduction="mean")

    for trial in range(5):
        pred   = torch.rand(2, 1, 64, 64)
        target = torch.rand(2, 1, 64, 64)

        our_psnr = psnr_fn(pred, target).item()
        ref_psnr = tm_psnr(pred, target, data_range=1.0).item()
        psnr_err = abs(our_psnr - ref_psnr)
        ok &= check(psnr_err < 0.05,
                    f"Trial {trial} PSNR: ours={our_psnr:.4f}, ref={ref_psnr:.4f}, "
                    f"err={psnr_err:.4f}")

        our_ssim = ssim_fn(pred, target).item()
        ref_ssim = tm_ssim(pred, target, data_range=1.0).item()
        ssim_err = abs(our_ssim - ref_ssim)
        ok &= check(ssim_err < 0.005,
                    f"Trial {trial} SSIM: ours={our_ssim:.4f}, ref={ref_ssim:.4f}, "
                    f"err={ssim_err:.4f}")

    return ok


# ===========================================================================
# Test 4: MetricTracker accumulation
# ===========================================================================

def test_metric_tracker():
    print("\n[4/7]  MetricTracker accumulation & best-epoch tracking...")

    ok = True
    tracker = MetricTracker(device=torch.device("cpu"))
    tracker.reset()

    # Simulate 3 batches of sizes 8, 8, 4 (last batch smaller)
    batch_configs = [(8, 0.05), (8, 0.08), (4, 0.12)]  # (size, noise_sigma)
    all_psnrs = []

    psnr_fn = PSNRMetric(data_range=1.0, reduction="none")

    for b, sigma in batch_configs:
        gt   = torch.rand(b, 1, 64, 64)
        pred = (gt + torch.randn_like(gt) * sigma).clamp(0, 1)
        tracker.update(pred, gt)

        # Manually compute expected PSNR for this batch
        per_img = psnr_fn(pred, gt)
        all_psnrs.extend(per_img.tolist())

    stats = tracker.compute()
    expected_psnr = sum(all_psnrs) / len(all_psnrs)  # unweighted mean over images
    psnr_err = abs(stats["psnr"] - expected_psnr)

    ok &= check(stats["n_images"] == 20,
                f"n_images = {stats['n_images']}  (expected 20)")
    ok &= check(psnr_err < 0.01,
                f"Accumulated PSNR = {stats['psnr']:.4f}, "
                f"expected = {expected_psnr:.4f}, err = {psnr_err:.4f}")
    ok &= check(0.0 < stats["ssim"] < 1.0,
                f"SSIM in (0,1): {stats['ssim']:.6f}")

    # -- Best-epoch tracking with controlled deterministic data
    # Use a clean GT and apply controlled noise levels to guarantee ordering
    gt_fixed    = torch.full((4, 1, 64, 64), 0.5)    # flat grey image
    pred_epoch1 = (gt_fixed + 0.01).clamp(0, 1)      # very close -> high PSNR/SSIM
    pred_epoch2 = (gt_fixed + 0.30).clamp(0, 1)      # far away  -> low  PSNR/SSIM

    # Epoch 1: high quality
    tracker.reset()
    tracker.update(pred_epoch1, gt_fixed)
    improved1 = tracker.update_best(epoch=1)
    ok &= check(improved1["psnr_improved"],
                "First epoch always improves PSNR (from -1.0)")
    ok &= check(tracker.best_epoch == 1,
                f"best_epoch = {tracker.best_epoch}  (expected 1)")

    # Epoch 2: low quality (definitely worse)
    tracker.reset()
    tracker.update(pred_epoch2, gt_fixed)
    improved2 = tracker.update_best(epoch=2)
    ok &= check(not improved2["psnr_improved"],
                "Worse epoch does NOT improve best PSNR")
    ok &= check(not improved2["ssim_improved"],
                "Worse epoch does NOT improve best SSIM")
    ok &= check(tracker.best_epoch == 1,
                f"best_epoch still = {tracker.best_epoch}  (expected 1)")

    since = tracker.epochs_since_improvement(current_epoch=2)
    ok &= check(since == 1,
                f"epochs_since_improvement(2) = {since}  (expected 1)")

    print(f"    {INFO}  {tracker.summary()}")
    return ok


# ===========================================================================
# Test 5: Denormalisation pipeline
# ===========================================================================

def test_denorm_pipeline():
    print("\n[5/7]  Denormalisation pipeline check...")

    ok = True

    # Simulate a model output in [-1, +1] and a GT in [-1, +1]
    pred_norm  = torch.randn(2, 1, 128, 128).clamp(-1, 1)
    gt_norm    = torch.randn(2, 1, 128, 128).clamp(-1, 1)

    # Denormalise to [0, 1]
    pred_01 = (pred_norm * 0.5 + 0.5).clamp(0, 1)
    gt_01   = (gt_norm   * 0.5 + 0.5).clamp(0, 1)

    ok &= check(pred_01.min() >= 0.0 and pred_01.max() <= 1.0,
                f"pred in [0,1]: [{pred_01.min():.4f}, {pred_01.max():.4f}]")
    ok &= check(gt_01.min() >= 0.0 and gt_01.max() <= 1.0,
                f"gt in [0,1]: [{gt_01.min():.4f}, {gt_01.max():.4f}]")

    psnr_fn = PSNRMetric(data_range=1.0)
    ssim_fn = SSIMMetric(channels=1, data_range=1.0)

    psnr_val = psnr_fn(pred_01, gt_01).item()
    ssim_val = ssim_fn(pred_01, gt_01).item()

    ok &= check(psnr_val > 0,   f"PSNR = {psnr_val:.2f} dB  (positive)")
    ok &= check(0 < ssim_val < 1, f"SSIM = {ssim_val:.4f}  (in (0,1))")

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

    psnr_fn = PSNRMetric(data_range=1.0).to(device)
    ssim_fn = SSIMMetric(channels=1, data_range=1.0).to(device)
    tracker = MetricTracker(device=device)

    pred   = torch.rand(4, 1, 64, 64, device=device)
    target = torch.rand(4, 1, 64, 64, device=device)

    psnr_val = psnr_fn(pred, target).item()
    ssim_val = ssim_fn(pred, target).item()

    ok &= check(psnr_val > 0,     f"CUDA PSNR = {psnr_val:.2f} dB")
    ok &= check(0 < ssim_val < 1, f"CUDA SSIM = {ssim_val:.4f}")

    tracker.reset()
    tracker.update(pred, target)
    stats = tracker.compute()
    ok &= check(stats["n_images"] == 4,
                f"Tracker n_images = {stats['n_images']}")

    return ok


# ===========================================================================
# Test 7: Visualisation (SSIM map on real image)
# ===========================================================================

def test_visualisation():
    print("\n[7/7]  Generating SSIM-map visualisation on real image pair...")

    gt_files = scan_directory(cfg.TRAIN_GT_DIR)
    lr_files = scan_directory(cfg.TRAIN_NOISYLR_DIR)

    if not gt_files:
        print(f"    {INFO}  No dataset files found -- skipping visualisation.")
        return True

    import cv2

    gt_raw = np.load(gt_files[0], allow_pickle=False)    # (256,256)
    lr_raw = np.load(lr_files[0], allow_pickle=False)    # (128,128)

    # Bilinear upscale LR to GT resolution for fair comparison
    lr_up = cv2.resize(lr_raw, (256, 256), interpolation=cv2.INTER_LINEAR)
    lr_up = np.clip(lr_up, 0, 1)

    # Convert to tensors in [0,1]
    gt_t  = torch.from_numpy(gt_raw[None, None, ...])
    lr_t  = torch.from_numpy(lr_up[None, None, ...])

    ssim_fn = SSIMMetric(channels=1, data_range=1.0)
    psnr_fn = PSNRMetric(data_range=1.0)

    ssim_score, ssim_map = ssim_fn.forward_with_map(lr_t, gt_t)
    psnr_val = psnr_fn(lr_t, gt_t).item()

    ssim_map_np = ssim_map.squeeze().numpy()
    diff_map    = np.abs(lr_up - gt_raw)

    fig = plt.figure(figsize=(20, 5), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.1,
                            left=0.03, right=0.97, top=0.88, bottom=0.03)

    def plot_img(ax, img, title, cmap="gray", vmin=0, vmax=1):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="#e6edf3", fontsize=9, pad=4)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_img(fig.add_subplot(gs[0, 0]), lr_up,
             f"NoisyLR (bilinear 2x)\n128->256")
    plot_img(fig.add_subplot(gs[0, 1]), gt_raw,
             "GT (ground truth)\n256x256")
    plot_img(fig.add_subplot(gs[0, 2]), diff_map,
             f"|LR - GT| (diff map)\nPSNR={psnr_val:.2f} dB", cmap="hot")
    plot_img(fig.add_subplot(gs[0, 3]), ssim_map_np,
             f"SSIM map (local)\nMean SSIM={ssim_score.item():.4f}",
             cmap="RdYlGn", vmin=0, vmax=1)
    plot_img(fig.add_subplot(gs[0, 4]),
             (ssim_map_np < 0.8).astype(np.float32),
             "Poor regions (SSIM < 0.8)\n(red = needs improvement)",
             cmap="hot", vmin=0, vmax=1)

    fig.suptitle(
        "AMSR-Net -- Phase 5: PSNR & SSIM Metrics on Real Dataset Sample\n"
        "Bilinear-upsampled NoisyLR vs GT  |  Lower SSIM = harder to restore",
        color="#e6edf3", fontsize=11, fontweight="bold", y=0.97
    )

    save_path = os.path.join(cfg.OUTPUTS_DIR, "phase5_metric_visualisation.png")
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [OK] Saved -> {save_path}")
    print(f"    {INFO}  Bilinear baseline: PSNR={psnr_val:.2f} dB, "
          f"SSIM={ssim_score.item():.4f}")
    print(f"    {INFO}  This is the FLOOR the model must beat to be useful.")
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  AMSR-Net | Phase 5 -- Metrics Verification")
    print("=" * 70)

    results = {
        "PSNRMetric":           test_psnr(),
        "SSIMMetric":           test_ssim(),
        "Cross-validation":     test_crossvalidate(),
        "MetricTracker":        test_metric_tracker(),
        "Denorm pipeline":      test_denorm_pipeline(),
        "GPU transfer":         test_gpu(),
        "Visualisation":        test_visualisation(),
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
        print("  All checks passed. Metrics are verified.")
        print("  Ready to proceed to Phase 6: Training Loop.")
    else:
        print("  Some checks FAILED. Fix errors above before Phase 6.")

    print("=" * 70)


if __name__ == "__main__":
    main()
