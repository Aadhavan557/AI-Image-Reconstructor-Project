"""
phase3_verify.py
================
AMSR-Net - Phase 3: Baseline CNN Verification
----------------------------------------------
Validates the BaselineCNN model end-to-end:

  1. Instantiation & architecture print.
  2. Parameter count check.
  3. Forward pass -- shapes, dtypes, output range.
  4. Backward pass -- gradient flow through all layers.
  5. GPU test     -- if CUDA is available.
  6. Throughput   -- images/sec on CPU and GPU.
  7. Feature map visualisation -- saves intermediate activations.

Run
---
  python phase3_verify.py
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# UTF-8 console fix for Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from models.baseline_cnn import (
    BaselineCNN, ResidualBlock, ChannelAttention, PixelShuffleUpsample
)
from dataset.semiconductor_dataset import normalise_lr, normalise_gt, denormalise

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

torch.manual_seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


def check(condition: bool, message: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"    {tag}  {message}")
    return condition


# ===========================================================================
# Test 1: Instantiation
# ===========================================================================

def test_instantiation():
    print("\n[1/7]  Model instantiation...")

    model = BaselineCNN(
        in_channels = cfg.CHANNELS,
        dim         = cfg.BASELINE_FEATURES,
        num_blocks  = cfg.BASELINE_NUM_BLOCKS,
        scale       = cfg.SCALE,
    )

    ok = True
    ok &= check(isinstance(model, nn.Module), "Model is nn.Module")
    ok &= check(model.scale == cfg.SCALE,     f"Scale = {model.scale}x")

    n_params = model.count_parameters()
    ok &= check(n_params > 0, f"Parameter count = {n_params:,}  ({n_params/1e6:.3f}M)")

    print(f"\n{model}")
    return ok, model


# ===========================================================================
# Test 2: Architecture sub-modules
# ===========================================================================

def test_architecture(model: BaselineCNN):
    print("\n[2/7]  Architecture sub-module checks...")

    ok = True
    ok &= check(hasattr(model, "shallow"),   "Has shallow feature extractor")
    ok &= check(hasattr(model, "body"),      "Has residual body")
    ok &= check(hasattr(model, "body_end"),  "Has body_end conv")
    ok &= check(hasattr(model, "upsample"),  "Has PixelShuffle upsample")
    ok &= check(hasattr(model, "head"),      "Has reconstruction head")

    # Check residual block count
    n_res_blocks = sum(1 for m in model.body if isinstance(m, ResidualBlock))
    ok &= check(n_res_blocks == cfg.BASELINE_NUM_BLOCKS,
                f"Residual blocks = {n_res_blocks} (expected {cfg.BASELINE_NUM_BLOCKS})")

    # Check PixelShuffle
    ups = model.upsample
    ok &= check(isinstance(ups, PixelShuffleUpsample),
                "Upsample is PixelShuffleUpsample")

    # Check channel attention in each residual block
    n_ca = sum(1 for m in model.modules() if isinstance(m, ChannelAttention))
    ok &= check(n_ca == cfg.BASELINE_NUM_BLOCKS,
                f"ChannelAttention blocks = {n_ca} (one per residual block)")

    return ok


# ===========================================================================
# Test 3: Forward pass shapes
# ===========================================================================

def test_forward_pass(model: BaselineCNN, device: torch.device):
    print(f"\n[3/7]  Forward pass (device={device})...")

    model.eval().to(device)

    ok = True
    test_cases = [
        (1,  cfg.PATCH_SIZE, cfg.PATCH_SIZE),    # single patch
        (4,  cfg.PATCH_SIZE, cfg.PATCH_SIZE),    # small batch
        (cfg.BATCH_SIZE, cfg.PATCH_SIZE, cfg.PATCH_SIZE),  # full batch
        (1,  128, 128),                           # full LR image
    ]

    with torch.no_grad():
        for b, h, w in test_cases:
            x = torch.randn(b, cfg.CHANNELS, h, w, device=device)
            y = model(x)
            exp_shape = (b, cfg.CHANNELS, h * cfg.SCALE, w * cfg.SCALE)
            ok &= check(
                tuple(y.shape) == exp_shape,
                f"Input {tuple(x.shape)} -> Output {tuple(y.shape)}  "
                f"(expected {exp_shape})"
            )
            ok &= check(
                y.dtype == torch.float32,
                f"Output dtype = {y.dtype}"
            )
            # tanh output should be in (-1, 1)
            y_min, y_max = y.min().item(), y.max().item()
            ok &= check(
                -1.01 <= y_min and y_max <= 1.01,
                f"Output range = [{y_min:.4f}, {y_max:.4f}]  (expected [-1, +1])"
            )

    return ok


# ===========================================================================
# Test 4: Backward pass / gradient flow
# ===========================================================================

def test_backward_pass(model: BaselineCNN, device: torch.device):
    print(f"\n[4/7]  Backward pass & gradient flow (device={device})...")

    model.train().to(device)

    x  = torch.randn(2, cfg.CHANNELS, cfg.PATCH_SIZE, cfg.PATCH_SIZE,
                     device=device, requires_grad=False)
    gt = torch.randn(2, cfg.CHANNELS,
                     cfg.PATCH_SIZE * cfg.SCALE, cfg.PATCH_SIZE * cfg.SCALE,
                     device=device)

    pred = model(x)
    loss = nn.functional.l1_loss(pred, gt)
    loss.backward()

    ok = True

    # Check that all parameters received a gradient
    no_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            no_grad.append(name)

    ok &= check(
        len(no_grad) == 0,
        f"All {model.count_parameters():,} params have gradients"
        if not no_grad else f"MISSING grads: {no_grad[:3]}"
    )

    # Check gradients are finite
    max_grad = max(p.grad.abs().max().item()
                   for p in model.parameters()
                   if p.grad is not None)
    ok &= check(
        np.isfinite(max_grad),
        f"Max gradient magnitude = {max_grad:.4e}  (finite)"
    )
    ok &= check(
        max_grad < 1e3,
        f"No gradient explosion  (max_grad={max_grad:.4e} < 1e3)"
    )

    print(f"    {INFO}  Loss value = {loss.item():.6f}")

    return ok


# ===========================================================================
# Test 5: GPU availability
# ===========================================================================

def test_gpu(model: BaselineCNN):
    print("\n[5/7]  GPU availability...")

    if not torch.cuda.is_available():
        print(f"    {INFO}  CUDA not available -- skipping GPU tests.")
        return True   # Not a failure -- GPU is optional for Phase 3

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

    ok = True
    ok &= check(True, f"GPU found: {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    # Quick forward pass on GPU
    model_gpu = BaselineCNN(
        in_channels=cfg.CHANNELS,
        dim=cfg.BASELINE_FEATURES,
        num_blocks=cfg.BASELINE_NUM_BLOCKS,
    ).to(device)

    with torch.no_grad():
        x = torch.randn(cfg.BATCH_SIZE, cfg.CHANNELS,
                        cfg.PATCH_SIZE, cfg.PATCH_SIZE, device=device)
        y = model_gpu(x)

    ok &= check(
        tuple(y.shape) == (cfg.BATCH_SIZE, cfg.CHANNELS,
                           cfg.PATCH_SIZE * cfg.SCALE,
                           cfg.PATCH_SIZE * cfg.SCALE),
        f"GPU forward pass OK: {tuple(x.shape)} -> {tuple(y.shape)}"
    )

    torch.cuda.empty_cache()
    return ok


# ===========================================================================
# Test 6: Throughput benchmark
# ===========================================================================

def test_throughput(model: BaselineCNN):
    print("\n[6/7]  Throughput benchmark...")

    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    ok = True
    WARMUP  = 5
    REPEATS = 20

    for device in devices_to_test:
        m = BaselineCNN(
            in_channels=cfg.CHANNELS,
            dim=cfg.BASELINE_FEATURES,
            num_blocks=cfg.BASELINE_NUM_BLOCKS,
        ).to(device).eval()

        x = torch.randn(cfg.BATCH_SIZE, cfg.CHANNELS,
                        cfg.PATCH_SIZE, cfg.PATCH_SIZE, device=device)

        with torch.no_grad():
            for _ in range(WARMUP):
                _ = m(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(REPEATS):
                _ = m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

        imgs_per_sec = cfg.BATCH_SIZE * REPEATS / elapsed
        ms_per_batch = elapsed / REPEATS * 1000

        print(f"    {INFO}  [{device.type.upper():4s}]  "
              f"{imgs_per_sec:.1f} img/s  |  {ms_per_batch:.1f} ms/batch  "
              f"(batch={cfg.BATCH_SIZE}, patch={cfg.PATCH_SIZE}x{cfg.PATCH_SIZE})")

        ok &= check(imgs_per_sec > 0, f"[{device.type}] Throughput > 0 img/s")

    return ok


# ===========================================================================
# Test 7: Feature map visualisation
# ===========================================================================

def test_feature_maps(model: BaselineCNN):
    print("\n[7/7]  Feature map visualisation...")

    import numpy as np
    from dataset.semiconductor_dataset import scan_directory

    gt_files = scan_directory(cfg.TRAIN_GT_DIR)
    lr_files = scan_directory(cfg.TRAIN_NOISYLR_DIR)

    if not gt_files:
        print(f"    {INFO}  No dataset files found -- skipping visualisation.")
        return True

    # Load a real sample
    lr_raw = np.load(lr_files[0], allow_pickle=False)   # (128, 128)
    gt_raw = np.load(gt_files[0], allow_pickle=False)   # (256, 256)

    from dataset.semiconductor_dataset import normalise_lr, normalise_gt
    lr_norm = torch.from_numpy(normalise_lr(lr_raw)[np.newaxis, np.newaxis, ...])
    gt_norm = torch.from_numpy(normalise_gt(gt_raw)[np.newaxis, np.newaxis, ...])

    model_cpu = BaselineCNN(
        in_channels=cfg.CHANNELS,
        dim=cfg.BASELINE_FEATURES,
        num_blocks=cfg.BASELINE_NUM_BLOCKS,
    ).eval()

    # Register hooks to capture intermediate feature maps
    feature_maps = {}

    def make_hook(name):
        def hook(module, inp, out):
            feature_maps[name] = out.detach().cpu()
        return hook

    model_cpu.shallow.register_forward_hook(make_hook("shallow"))
    model_cpu.body[-1].register_forward_hook(make_hook("last_res_block"))
    model_cpu.upsample.register_forward_hook(make_hook("after_upsample"))

    with torch.no_grad():
        pred = model_cpu(lr_norm)

    # Convert tensors back to displayable images
    lr_display  = np.clip(lr_norm.squeeze().numpy() * 0.5 + 0.5, 0, 1)
    gt_display  = np.clip(gt_norm.squeeze().numpy() * 0.5 + 0.5, 0, 1)
    out_display = np.clip(pred.squeeze().numpy()    * 0.5 + 0.5, 0, 1)

    n_fmap_show = 8   # Show first 8 feature channels

    fig = plt.figure(figsize=(22, 14), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, n_fmap_show + 2, figure=fig,
                            hspace=0.35, wspace=0.08,
                            left=0.03, right=0.97, top=0.93, bottom=0.03)

    def show_img(ax, img, title, cmap="gray"):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, color="#e6edf3", fontsize=7, pad=3)
        ax.axis("off")

    # Row 0: Input | GT | Output
    show_img(fig.add_subplot(gs[0, 0]), lr_display,
             f"Input NoisyLR\n128x128")
    show_img(fig.add_subplot(gs[0, 1]), gt_display,
             f"GT (ground truth)\n256x256")
    show_img(fig.add_subplot(gs[0, 2]), out_display,
             f"Model Output\n256x256 (untrained)")

    for col in range(3, n_fmap_show + 2):
        fig.add_subplot(gs[0, col]).axis("off")

    # Rows 1-3: Feature maps at each stage
    stage_labels = ["shallow", "last_res_block", "after_upsample"]
    stage_names  = [
        f"Shallow Features\n(dim={cfg.BASELINE_FEATURES}, 128x128)",
        f"Last Residual Block\n(dim={cfg.BASELINE_FEATURES}, 128x128)",
        f"After PixelShuffle\n(dim={cfg.BASELINE_FEATURES}, 256x256)",
    ]

    for row, (key, name) in enumerate(zip(stage_labels, stage_names)):
        fmap = feature_maps[key].squeeze(0)   # (C, H, W)
        n_show = min(n_fmap_show, fmap.shape[0])

        # Normalise each feature map independently to [0, 1] for display
        for col in range(n_show):
            fm = fmap[col].numpy()
            fm_min, fm_max = fm.min(), fm.max()
            fm_norm = (fm - fm_min) / (fm_max - fm_min + 1e-8)
            ax = fig.add_subplot(gs[row + 1, col])
            ax.imshow(fm_norm, cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"ch {col}", color="#8b949e", fontsize=6, pad=2)
            ax.axis("off")

        for col in range(n_show, n_fmap_show + 2):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.set_facecolor("#0d1117")
            ax.axis("off")

        # Label the row
        fig.text(0.01, 0.76 - row * 0.22, name,
                 color="#e6edf3", fontsize=8, va="center", rotation=90)

    fig.suptitle(
        "AMSR-Net -- Phase 3: BaselineCNN Feature Maps\n"
        f"Untrained model  |  dim={cfg.BASELINE_FEATURES}  |  "
        f"blocks={cfg.BASELINE_NUM_BLOCKS}  |  scale=2x",
        color="#e6edf3", fontsize=12, fontweight="bold", y=0.97
    )

    save_path = os.path.join(cfg.OUTPUTS_DIR, "phase3_feature_maps.png")
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [OK] Feature maps saved -> {save_path}")
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  AMSR-Net | Phase 3 -- Baseline CNN Verification")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    results = {}

    ok, model = test_instantiation()
    results["Instantiation"]   = ok
    results["Architecture"]    = test_architecture(model)
    results["Forward pass"]    = test_forward_pass(model, device)
    results["Backward pass"]   = test_backward_pass(model, device)
    results["GPU test"]        = test_gpu(model)
    results["Throughput"]      = test_throughput(model)
    results["Feature maps"]    = test_feature_maps(model)

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
        print("  All checks passed. BaselineCNN is verified.")
        print("  Ready to proceed to Phase 4: Loss Functions.")
    else:
        print("  Some checks FAILED. Fix errors above before Phase 4.")

    print("=" * 70)


if __name__ == "__main__":
    main()
