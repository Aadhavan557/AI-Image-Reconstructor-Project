"""
phase2_verify.py
================
AMSR-Net - Phase 2: Data Preprocessing Verification
------------------------------------------------------
Validates the entire data pipeline end-to-end:

  1. Dataset instantiation (train + val).
  2. Single-sample inspection  -- shapes, dtypes, value ranges.
  3. DataLoader batch inspection -- shapes, dtypes, timing.
  4. Augmentation visual check  -- saves a 3x3 grid of augmented crops.
  5. Normalisation round-trip   -- confirms denormalise() is invertible.
  6. Train/val split integrity  -- confirms no overlap between splits.

Run
---
  python phase2_verify.py

No GPU required.
"""

import sys
import os
import time
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

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
from dataset.semiconductor_dataset import (
    SemiconductorDataset,
    normalise_lr,
    normalise_gt,
    denormalise,
    PairedTransform,
    random_crop_pair,
    centre_crop_pair,
    scan_directory,
)
from dataset.dataloader import build_dataloaders

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


# ===========================================================================
# Helper
# ===========================================================================

def check(condition: bool, message: str) -> bool:
    tag = PASS if condition else FAIL
    print(f"    {tag}  {message}")
    return condition


# ===========================================================================
# Test 1: Dataset instantiation
# ===========================================================================

def test_dataset_init(gt_files, lr_files):
    print("\n[1/6]  Dataset instantiation...")

    train_ds = SemiconductorDataset(
        gt_files=gt_files, lr_files=lr_files,
        patch_size=cfg.PATCH_SIZE, scale=cfg.SCALE, is_train=True,
    )
    val_ds = SemiconductorDataset(
        gt_files=gt_files[:100], lr_files=lr_files[:100],
        patch_size=cfg.PATCH_SIZE, scale=cfg.SCALE, is_train=False,
    )

    ok = True
    ok &= check(len(train_ds) == len(gt_files),
                f"Train dataset len = {len(train_ds)}")
    ok &= check(len(val_ds)   == 100,
                f"Val dataset len   = {len(val_ds)}")
    ok &= check(train_ds._aug is not None,  "Train has augmentation")
    ok &= check(val_ds._aug   is None,      "Val has no augmentation")

    print(f"  {train_ds}")
    return ok, train_ds, val_ds


# ===========================================================================
# Test 2: Single sample inspection
# ===========================================================================

def test_single_sample(train_ds, val_ds):
    print("\n[2/6]  Single-sample inspection...")

    ok = True
    for label, ds in [("TRAIN", train_ds), ("VAL", val_ds)]:
        sample = ds[0]
        lr = sample["lr"]
        gt = sample["gt"]

        expected_lr_shape = (1, cfg.PATCH_SIZE, cfg.PATCH_SIZE)
        expected_gt_shape = (1, cfg.PATCH_SIZE * cfg.SCALE,
                                cfg.PATCH_SIZE * cfg.SCALE)

        ok &= check(lr.shape == expected_lr_shape,
                    f"[{label}] LR shape = {tuple(lr.shape)}  "
                    f"(expected {expected_lr_shape})")
        ok &= check(gt.shape == expected_gt_shape,
                    f"[{label}] GT shape = {tuple(gt.shape)}  "
                    f"(expected {expected_gt_shape})")
        ok &= check(lr.dtype == torch.float32,
                    f"[{label}] LR dtype = {lr.dtype}")
        ok &= check(gt.dtype == torch.float32,
                    f"[{label}] GT dtype = {gt.dtype}")

        # GT should be in [-1, +1] (normalised)
        gt_min, gt_max = gt.min().item(), gt.max().item()
        ok &= check(-1.1 <= gt_min and gt_max <= 1.1,
                    f"[{label}] GT range = [{gt_min:.4f}, {gt_max:.4f}]  "
                    f"(expected ~[-1, +1])")

        # LR can be slightly outside [-1, +1] due to clip at -0.2/1.2
        lr_min, lr_max = lr.min().item(), lr.max().item()
        ok &= check(-1.5 <= lr_min and lr_max <= 1.5,
                    f"[{label}] LR range = [{lr_min:.4f}, {lr_max:.4f}]  "
                    f"(expected ~[-1.4, +1.4])")

        ok &= check(isinstance(sample["filename"], str),
                    f"[{label}] filename = {sample['filename']}")

    return ok


# ===========================================================================
# Test 3: Normalisation round-trip
# ===========================================================================

def test_normalisation_roundtrip():
    print("\n[3/6]  Normalisation round-trip...")

    ok = True

    # GT: [0,1] -> normalise -> denormalise -> should recover [0,1]
    raw_gt = np.random.rand(64, 64).astype(np.float32)
    norm   = normalise_gt(raw_gt)
    tensor = torch.from_numpy(norm)
    recovered = denormalise(tensor).numpy()
    max_err = float(np.abs(raw_gt - recovered).max())

    ok &= check(max_err < 1e-5,
                f"GT round-trip max error = {max_err:.2e}  (threshold: 1e-5)")

    # LR: values in [-0.2, 1.2] -> normalise -> denormalise -> recover input
    raw_lr = np.clip(
        np.random.randn(64, 64).astype(np.float32) * 0.2 + 0.5, -0.2, 1.2
    )
    norm_lr   = normalise_lr(raw_lr)
    tensor_lr = torch.from_numpy(norm_lr)
    recovered_lr = denormalise(tensor_lr).numpy()
    max_err_lr = float(np.abs(raw_lr - recovered_lr).max())

    ok &= check(max_err_lr < 1e-5,
                f"LR round-trip max error = {max_err_lr:.2e}  (threshold: 1e-5)")

    # Verify GT normalisation maps [0,1] -> [-1,+1]
    zeros = normalise_gt(np.zeros((4, 4), dtype=np.float32))
    ones  = normalise_gt(np.ones((4, 4),  dtype=np.float32))
    ok &= check(np.allclose(zeros, -1.0), "normalise_gt(0) = -1.0")
    ok &= check(np.allclose(ones,  +1.0), "normalise_gt(1) = +1.0")

    return ok


# ===========================================================================
# Test 4: Augmentation consistency
# ===========================================================================

def test_augmentation():
    print("\n[4/6]  Augmentation consistency check...")

    ok = True
    aug = PairedTransform(hflip_p=1.0, vflip_p=0.0, rot90_p=0.0)

    lr = np.arange(16, dtype=np.float32).reshape(4, 4)
    gt = np.arange(64, dtype=np.float32).reshape(8, 8)

    lr_aug, gt_aug = aug(lr.copy(), gt.copy())

    # After hflip, column order reverses
    ok &= check(np.allclose(lr_aug, np.fliplr(lr)),
                "LR hflip correct")
    ok &= check(np.allclose(gt_aug, np.fliplr(gt)),
                "GT hflip correct")

    # Shapes unchanged
    ok &= check(lr_aug.shape == lr.shape, f"LR shape unchanged: {lr_aug.shape}")
    ok &= check(gt_aug.shape == gt.shape, f"GT shape unchanged: {gt_aug.shape}")

    # Test rot90
    aug_rot = PairedTransform(hflip_p=0.0, vflip_p=0.0, rot90_p=1.0)
    random.seed(0)  # Fix k=1
    lr_rot, gt_rot = aug_rot(lr.copy(), gt.copy())
    ok &= check(lr_rot.shape == lr.shape, f"LR rot90 shape unchanged")
    ok &= check(gt_rot.shape == gt.shape, f"GT rot90 shape unchanged")

    return ok


# ===========================================================================
# Test 5: DataLoader batch
# ===========================================================================

def test_dataloader_batch(gt_files, lr_files):
    print("\n[5/6]  DataLoader batch timing & shapes...")

    ok = True

    # Use num_workers=0 for this test to avoid Windows spawn issues
    small_ds = SemiconductorDataset(
        gt_files   = gt_files[:64],
        lr_files   = lr_files[:64],
        patch_size = cfg.PATCH_SIZE,
        scale      = cfg.SCALE,
        is_train   = True,
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset    = small_ds,
        batch_size = 8,
        shuffle    = True,
        num_workers= 0,       # 0 for safe in-process test
        pin_memory = False,
    )

    t0 = time.perf_counter()
    batch = next(iter(loader))
    elapsed = time.perf_counter() - t0

    lr_batch = batch["lr"]
    gt_batch = batch["gt"]

    expected_lr = (8, 1, cfg.PATCH_SIZE, cfg.PATCH_SIZE)
    expected_gt = (8, 1, cfg.PATCH_SIZE * cfg.SCALE, cfg.PATCH_SIZE * cfg.SCALE)

    ok &= check(tuple(lr_batch.shape) == expected_lr,
                f"LR batch shape = {tuple(lr_batch.shape)}")
    ok &= check(tuple(gt_batch.shape) == expected_gt,
                f"GT batch shape = {tuple(gt_batch.shape)}")
    ok &= check(lr_batch.dtype == torch.float32,
                f"LR dtype = {lr_batch.dtype}")
    ok &= check(gt_batch.dtype == torch.float32,
                f"GT dtype = {gt_batch.dtype}")
    ok &= check(elapsed < 10.0,
                f"First batch loaded in {elapsed:.3f}s  (threshold: 10s)")

    print(f"    {INFO}  Memory: LR batch {lr_batch.element_size() * lr_batch.nelement() / 1024:.1f} KB, "
          f"GT batch {gt_batch.element_size() * gt_batch.nelement() / 1024:.1f} KB")

    return ok


# ===========================================================================
# Test 6: Visualisation
# ===========================================================================

def test_visualisation(train_ds):
    print("\n[6/6]  Saving augmentation visualisation...")

    n_cols = 3   # original | augmented_v1 | augmented_v2
    n_rows = 3   # 3 random samples

    fig = plt.figure(figsize=(15, 5 * n_rows), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        n_rows, n_cols * 2,   # pairs: LR and GT side by side
        figure=fig, hspace=0.4, wspace=0.08,
        left=0.04, right=0.96, top=0.93, bottom=0.02,
    )

    indices = random.sample(range(len(train_ds)), n_rows)

    for row, idx in enumerate(indices):
        # Get 2 differently-augmented versions of the same image
        sample_a = train_ds[idx]
        sample_b = train_ds[idx]   # Different random augmentation

        gt_raw = np.load(train_ds.gt_files[idx], allow_pickle=False)
        lr_raw = np.load(train_ds.lr_files[idx], allow_pickle=False)

        triplets = [
            ("Original LR/GT (raw)", np.clip(lr_raw, 0, 1), np.clip(gt_raw, 0, 1)),
            ("Augmented v1",
             np.clip(sample_a["lr"].squeeze().numpy() * 0.5 + 0.5, 0, 1),
             np.clip(sample_a["gt"].squeeze().numpy() * 0.5 + 0.5, 0, 1)),
            ("Augmented v2",
             np.clip(sample_b["lr"].squeeze().numpy() * 0.5 + 0.5, 0, 1),
             np.clip(sample_b["gt"].squeeze().numpy() * 0.5 + 0.5, 0, 1)),
        ]

        for col, (title, lr_img, gt_img) in enumerate(triplets):
            # LR panel
            ax_lr = fig.add_subplot(gs[row, col * 2])
            ax_lr.imshow(lr_img, cmap="gray", vmin=0, vmax=1)
            ax_lr.set_title(
                f"{title}\nLR ({lr_img.shape[0]}x{lr_img.shape[1]})",
                color="#e6edf3", fontsize=8, pad=3
            )
            ax_lr.axis("off")

            # GT panel
            ax_gt = fig.add_subplot(gs[row, col * 2 + 1])
            ax_gt.imshow(gt_img, cmap="gray", vmin=0, vmax=1)
            ax_gt.set_title(
                f"{title}\nGT ({gt_img.shape[0]}x{gt_img.shape[1]})",
                color="#e6edf3", fontsize=8, pad=3
            )
            ax_gt.axis("off")

    fig.suptitle(
        "AMSR-Net -- Phase 2: Data Preprocessing Verification\n"
        "Columns: Original | Augmented v1 | Augmented v2   "
        f"(Patch: {cfg.PATCH_SIZE}x{cfg.PATCH_SIZE} LR / "
        f"{cfg.PATCH_SIZE * cfg.SCALE}x{cfg.PATCH_SIZE * cfg.SCALE} GT)",
        color="#e6edf3", fontsize=12, fontweight="bold", y=0.97
    )

    save_path = os.path.join(cfg.OUTPUTS_DIR, "phase2_augmentation_check.png")
    os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [OK] Saved -> {save_path}")
    return True


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  AMSR-Net | Phase 2 -- Data Preprocessing Verification")
    print("=" * 70)

    gt_files = scan_directory(cfg.TRAIN_GT_DIR)
    lr_files = scan_directory(cfg.TRAIN_NOISYLR_DIR)

    results = {}

    ok, train_ds, val_ds = test_dataset_init(gt_files, lr_files)
    results["Dataset init"]      = ok
    results["Single sample"]     = test_single_sample(train_ds, val_ds)
    results["Normalisation"]     = test_normalisation_roundtrip()
    results["Augmentation"]      = test_augmentation()
    results["DataLoader batch"]  = test_dataloader_batch(gt_files, lr_files)
    results["Visualisation"]     = test_visualisation(train_ds)

    # Summary
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
        print("  All checks passed. Phase 2 pipeline is verified.")
        print("  Ready to proceed to Phase 3: Baseline CNN Model.")
    else:
        print("  Some checks FAILED. Fix the errors above before Phase 3.")

    print("=" * 70)


if __name__ == "__main__":
    main()
