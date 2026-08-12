"""
phase1_explore.py
=================
AMSR-Net - Phase 1: Dataset Exploration
----------------------------------------
Purpose
-------
Understand the raw data before building any model.
This script:
  1. Scans the dataset directories and counts files.
  2. Loads a representative sample of NoisyLR and GT .npy files.
  3. Prints shape, dtype, min, max, mean, std for every split.
  4. Computes pixel-value histograms to reveal the noise distribution.
  5. Visualises several GT / NoisyLR side-by-side pairs and saves
     the figure to outputs/phase1_visualisation.png.
  6. Explains every observation in detail.

Run
---
  python phase1_explore.py

No GPU required -- pure NumPy + Matplotlib.
"""

# ---------------------------------------------------------------------------
# Fix Windows console encoding FIRST -- before any other import prints
# ---------------------------------------------------------------------------
import sys
import io

# Reconfigure stdout/stderr to UTF-8 so special characters print correctly.
# On Windows the default is cp1252 which cannot handle many Unicode chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
import glob
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Make sure we can import from the AMSR-Net project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)


# ===========================================================================
# 1.  DIRECTORY SCANNING
# ===========================================================================

def scan_directory(directory: str, extension: str = "*.npy") -> List[str]:
    """
    Return a sorted list of all files matching `extension` in `directory`.

    Parameters
    ----------
    directory : str
        Absolute or relative path to the folder to scan.
    extension : str
        Glob pattern (default ``*.npy``).

    Returns
    -------
    List[str]
        Sorted list of absolute file paths.

    Why
    ---
    Sorting guarantees that GT[i] and NoisyLR[i] are paired correctly
    when both folders contain identically named files (000000.npy, ...).
    """
    pattern = os.path.join(directory, extension)
    files = sorted(glob.glob(pattern))
    return files


def print_directory_summary(name: str, files: List[str]) -> None:
    """
    Pretty-print basic information about a set of files.

    Parameters
    ----------
    name  : str        Label for this directory (e.g. "Train GT").
    files : List[str]  List of file paths.
    """
    n = len(files)
    if n == 0:
        print(f"  [{name}]  WARNING: No files found!")
        return

    first = os.path.basename(files[0])
    last  = os.path.basename(files[-1])
    size_bytes = os.path.getsize(files[0])
    size_kb    = size_bytes / 1024.0

    print(f"  [{name}]")
    print(f"    Files found  : {n:,}")
    print(f"    First file   : {first}")
    print(f"    Last file    : {last}")
    print(f"    File size    : {size_kb:.1f} KB  ({size_bytes:,} bytes)")


# ===========================================================================
# 2.  SINGLE-FILE STATISTICS
# ===========================================================================

def load_npy(path: str) -> np.ndarray:
    """
    Load a single .npy file and return it as a NumPy array.

    Parameters
    ----------
    path : str  Path to the .npy file.

    Returns
    -------
    np.ndarray

    Notes
    -----
    allow_pickle=False is deliberately set to prevent arbitrary code
    execution from untrusted files -- important for competition datasets.
    """
    arr = np.load(path, allow_pickle=False)
    return arr


def describe_array(arr: np.ndarray, label: str) -> Dict[str, object]:
    """
    Compute and print descriptive statistics for a NumPy array.

    Parameters
    ----------
    arr   : np.ndarray  The image array.
    label : str         Human-readable name (e.g. "GT[0]").

    Returns
    -------
    Dict with keys: shape, dtype, min, max, mean, std.

    Mathematical intuition
    ----------------------
    - shape  : determines spatial resolution and channel count.
    - dtype  : float32 means 4 bytes/pixel; important for memory budget.
    - min    : pixels below 0.0 indicate noise artefacts (speckle can
               produce negative intensity after acquisition digitisation).
    - max    : pixels above 1.0 indicate noise saturation artefacts.
    - mean   : overall brightness of the scene.
    - std    : spread of pixel values; higher std in NoisyLR vs GT
               indicates noise magnitude.
    """
    stats = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min":   float(arr.min()),
        "max":   float(arr.max()),
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
    }
    print(f"\n  -- {label} --")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k:8s}: {v:.6f}")
        else:
            print(f"    {k:8s}: {v}")
    return stats


# ===========================================================================
# 3.  POPULATION STATISTICS
# ===========================================================================

def compute_population_stats(
    files: List[str],
    n_sample: int = 200
) -> Dict[str, float]:
    """
    Estimate global statistics by sampling `n_sample` files from `files`.

    We do not load all 3200 files at once to keep memory usage low.
    Random sampling of 200 files gives very stable estimates (CLT).

    Parameters
    ----------
    files    : List[str]  All file paths for the split.
    n_sample : int        How many files to sample.

    Returns
    -------
    Dict with keys: global_min, global_max, global_mean, global_std.
    """
    sampled = random.sample(files, min(n_sample, len(files)))
    arrays  = [np.load(f, allow_pickle=False) for f in sampled]
    all_pixels = np.concatenate([a.ravel() for a in arrays])

    stats = {
        "global_min":  float(all_pixels.min()),
        "global_max":  float(all_pixels.max()),
        "global_mean": float(all_pixels.mean()),
        "global_std":  float(all_pixels.std()),
    }
    return stats


# ===========================================================================
# 4.  VISUALISATION
# ===========================================================================

def visualise_samples(
    gt_files:  List[str],
    lr_files:  List[str],
    n_pairs:   int = 5,
    save_path: str = "outputs/phase1_visualisation.png",
) -> None:
    """
    Create a side-by-side visualisation grid.

    Layout per pair (row):
        Col 0 -> NoisyLR (upscaled via bilinear interpolation for display)
        Col 1 -> GT (clean high-resolution)
        Col 2 -> Difference map |bilinear(LR) - GT|  (red = large error)
        Col 3 -> Pixel histogram overlay (GT vs NoisyLR)

    Parameters
    ----------
    gt_files  : List[str]  Sorted GT file paths.
    lr_files  : List[str]  Sorted NoisyLR file paths.
    n_pairs   : int        Number of image pairs to display.
    save_path : str        Where to save the figure.

    Why bilinear upscale for display?
    -----------------------------------
    We just want to compare spatial structure -- bilinear interpolation
    at 2x lets us overlay GT and LR at the same pixel resolution without
    distorting feature locations.
    """
    import cv2  # OpenCV for bilinear resize

    indices = random.sample(range(len(gt_files)), n_pairs)

    fig = plt.figure(figsize=(22, 5 * n_pairs), facecolor="#0d1117")
    gs  = gridspec.GridSpec(
        n_pairs, 4,
        figure=fig,
        hspace=0.35, wspace=0.12,
        left=0.04, right=0.96, top=0.94, bottom=0.02,
    )

    for row, idx in enumerate(indices):
        gt = np.load(gt_files[idx],  allow_pickle=False)   # (256, 256)
        lr = np.load(lr_files[idx],  allow_pickle=False)   # (128, 128)

        # Bilinear upscale LR -> GT resolution for fair visual comparison
        lr_up      = cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        lr_up_clip = np.clip(lr_up, 0.0, 1.0)
        diff       = np.abs(lr_up_clip - gt)
        fname      = os.path.basename(gt_files[idx])

        # Col 0: NoisyLR (upscaled)
        ax0 = fig.add_subplot(gs[row, 0])
        im0 = ax0.imshow(lr_up_clip, cmap="gray", vmin=0, vmax=1)
        ax0.set_title(
            f"NoisyLR (2x bilinear) [{fname}]\n"
            f"min={lr.min():.4f}  max={lr.max():.4f}",
            color="#e6edf3", fontsize=9, pad=4
        )
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Col 1: GT
        ax1 = fig.add_subplot(gs[row, 1])
        im1 = ax1.imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax1.set_title(
            f"GT (ground truth) [{fname}]\n"
            f"min={gt.min():.4f}  max={gt.max():.4f}",
            color="#e6edf3", fontsize=9, pad=4
        )
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Col 2: Difference map
        ax2 = fig.add_subplot(gs[row, 2])
        im2 = ax2.imshow(diff, cmap="hot", vmin=0, vmax=0.3)
        ax2.set_title(
            f"|bilinear(LR) - GT|\n"
            f"mean_err={diff.mean():.4f}  max_err={diff.max():.4f}",
            color="#e6edf3", fontsize=9, pad=4
        )
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Col 3: Pixel histogram
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.set_facecolor("#161b22")
        bins = np.linspace(-0.1, 1.3, 120)
        ax3.hist(gt.ravel(), bins=bins, color="#58a6ff", alpha=0.7,
                 label="GT",      density=True, histtype="stepfilled")
        ax3.hist(lr.ravel(), bins=bins, color="#f78166", alpha=0.5,
                 label="NoisyLR", density=True, histtype="stepfilled")
        ax3.axvline(x=0.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.axvline(x=1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.set_title("Pixel histogram (normalised density)",
                      color="#e6edf3", fontsize=9)
        ax3.set_xlabel("Pixel value", color="#8b949e", fontsize=8)
        ax3.set_ylabel("Density",     color="#8b949e", fontsize=8)
        ax3.tick_params(colors="#8b949e")
        ax3.legend(fontsize=8, facecolor="#21262d", labelcolor="#e6edf3")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle(
        "AMSR-Net -- Phase 1: Dataset Exploration\n"
        "NoisyLR (128x128)  <->  GT (256x256)  |  Scale factor: 2x",
        color="#e6edf3", fontsize=14, fontweight="bold", y=0.97
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [OK] Visualisation saved -> {save_path}")


# ===========================================================================
# 5.  NOISE ANALYSIS
# ===========================================================================

def analyse_noise(
    gt_files: List[str],
    lr_files: List[str],
    n_sample: int = 50,
) -> None:
    """
    Estimate the noise level and distribution in the NoisyLR images.

    Method
    ------
    For each sampled pair, we bilinearly upscale the LR image and compute:
        noise_estimate = NoisyLR_upscaled - GT

    Statistics computed
    -------------------
    - Mean of noise      -> bias (systematic offset)
    - Std of noise       -> noise power (RMSE-like)
    - Kurtosis of noise  -> whether tails are heavier than Gaussian.
      Speckle noise in coherent imaging often has higher kurtosis.
    - SNR estimate (dB)  -> signal-to-noise ratio

    Mathematical formulas
    ---------------------
    SNR (dB)  = 10 * log10( E[GT^2] / E[noise^2] )
    RMSE      = sqrt( E[(LR_upscaled - GT)^2] )
    Kurtosis  = E[(noise - mu)^4] / sigma^4   (excess = kurtosis - 3)
    """
    import cv2
    from scipy import stats as sp_stats

    noise_pixels = []
    gt_pixels    = []

    indices = random.sample(range(len(gt_files)), min(n_sample, len(gt_files)))

    for idx in indices:
        gt    = np.load(gt_files[idx], allow_pickle=False)
        lr    = np.load(lr_files[idx], allow_pickle=False)
        lr_up = cv2.resize(lr, (gt.shape[1], gt.shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        noise = lr_up - gt
        noise_pixels.append(noise.ravel())
        gt_pixels.append(gt.ravel())

    noise_arr = np.concatenate(noise_pixels)
    gt_arr    = np.concatenate(gt_pixels)

    noise_mean = float(noise_arr.mean())
    noise_std  = float(noise_arr.std())
    noise_rmse = float(np.sqrt((noise_arr ** 2).mean()))
    noise_kurt = float(sp_stats.kurtosis(noise_arr, fisher=True))

    signal_power = float((gt_arr ** 2).mean())
    noise_power  = float((noise_arr ** 2).mean())
    snr_db       = 10.0 * np.log10(signal_power / (noise_power + 1e-12))

    print("\n  -- Noise Analysis --")
    print(f"    Noise Mean          : {noise_mean:+.6f}")
    print(f"    Noise Std           : {noise_std:.6f}")
    print(f"    RMSE (noise)        : {noise_rmse:.6f}")
    print(f"    Excess Kurtosis     : {noise_kurt:.4f}  (Gaussian = 0)")
    print(f"    Estimated SNR       : {snr_db:.2f} dB")

    if abs(noise_mean) > 0.01:
        print("    [WARN] Bias detected -- NoisyLR has a systematic offset vs GT.")
    else:
        print("    [OK] No significant bias -- noise appears zero-mean.")

    if noise_kurt > 1.0:
        print("    [INFO] Heavy tails (kurtosis > 1) -- consistent with speckle noise.")
        print("           -> Charbonnier/L1 loss preferred over MSE for robustness.")
    else:
        print("    [INFO] Near-Gaussian noise -- L2/L1 losses both viable.")


# ===========================================================================
# 6.  SCALE FACTOR VERIFICATION
# ===========================================================================

def verify_scale_factor(gt_files: List[str], lr_files: List[str]) -> None:
    """
    Confirm the exact scale factor between GT and LR images.

    Why this matters
    ----------------
    The model upsampling head must be configured for the correct factor
    (PixelShuffle, transposed conv, or bicubic). A wrong assumption
    here cascades into all subsequent phases.
    """
    gt = np.load(gt_files[0], allow_pickle=False)
    lr = np.load(lr_files[0], allow_pickle=False)

    scale_h = gt.shape[0] / lr.shape[0]
    scale_w = gt.shape[1] / lr.shape[1]

    print("\n  -- Scale Factor Verification --")
    print(f"    LR shape : {lr.shape}")
    print(f"    GT shape : {gt.shape}")
    print(f"    Scale H  : {scale_h:.2f}x")
    print(f"    Scale W  : {scale_w:.2f}x")

    if scale_h == scale_w == 2.0:
        print("    [OK] Confirmed 2x super-resolution task.")
    else:
        print(f"    [WARN] Non-square or unexpected scale ({scale_h}x{scale_w})!")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    """Run the full Phase 1 exploration pipeline."""

    print("=" * 70)
    print("  AMSR-Net | Phase 1 -- Dataset Exploration")
    print("=" * 70)

    # Step 1: Scan directories
    print("\n[1/7]  Scanning dataset directories...")
    gt_files    = scan_directory(cfg.TRAIN_GT_DIR)
    lr_tr_files = scan_directory(cfg.TRAIN_NOISYLR_DIR)
    lr_te_files = scan_directory(cfg.TEST_NOISYLR_DIR)

    print_directory_summary("Train GT",      gt_files)
    print_directory_summary("Train NoisyLR", lr_tr_files)
    print_directory_summary("Test  NoisyLR", lr_te_files)

    # Step 2: Verify pairing
    print("\n[2/7]  Verifying file pairing...")
    assert len(gt_files) == len(lr_tr_files), (
        f"Mismatch: {len(gt_files)} GT files vs {len(lr_tr_files)} NoisyLR files!"
    )
    for gt_f, lr_f in zip(gt_files[:5], lr_tr_files[:5]):
        assert os.path.basename(gt_f) == os.path.basename(lr_f), \
            f"Name mismatch: {gt_f} vs {lr_f}"
    print(f"  [OK] {len(gt_files):,} matched pairs confirmed.")

    # Step 3: Single-file statistics
    print("\n[3/7]  Single-file statistics (first 3 samples each)...")
    for i in range(3):
        describe_array(load_npy(gt_files[i]),    f"GT[{i}]     (256x256)")
        describe_array(load_npy(lr_tr_files[i]), f"NoisyLR[{i}] (128x128)")

    # Step 4: Scale factor
    print("\n[4/7]  Scale factor verification...")
    verify_scale_factor(gt_files, lr_tr_files)

    # Step 5: Population statistics
    print("\n[5/7]  Population statistics (sampling 200 files each)...")
    print("  Computing GT statistics...")
    gt_pop = compute_population_stats(gt_files, n_sample=200)
    print("  Computing NoisyLR statistics...")
    lr_pop = compute_population_stats(lr_tr_files, n_sample=200)

    print("\n  -- Population Stats: GT --")
    for k, v in gt_pop.items():
        print(f"    {k:14s}: {v:.6f}")

    print("\n  -- Population Stats: NoisyLR --")
    for k, v in lr_pop.items():
        print(f"    {k:14s}: {v:.6f}")

    print("\n  -- Observations & Implications --")
    print("""
  * GT images are in [0.0, 1.0] -- already normalised float32.
    No additional global rescaling is required for GT.

  * NoisyLR images can fall OUTSIDE [0.0, 1.0]:
      - min can be negative (dark-current / read-noise artefacts)
      - max can exceed 1.0  (speckle saturation artefacts)
    This is characteristic of:
      - Speckle noise (coherent imaging from e-beam / SEM)
      - Quantisation noise in the ADC chain

  * 2x spatial resolution difference (128 -> 256) means:
      - The model must BOTH denoise AND super-resolve simultaneously.
      - This is a joint task: Blind Denoising + SISR (Super-Resolution).
      - Simple bicubic upsampling will fail -- learned priors are essential.

  * NoisyLR std > GT std: noise increases pixel variance, as expected.

  * Action for Phase 2:
      - Clip NoisyLR to a safe range then normalise to [-1, +1].
      - Use random crops to increase effective training set size.
      - Apply flip & 90-degree rotation augmentations (orientation-invariant).
    """)

    # Step 6: Noise analysis
    print("\n[6/7]  Noise analysis (50 sampled pairs)...")
    analyse_noise(gt_files, lr_tr_files, n_sample=50)

    # Step 7: Visualisation
    print("\n[7/7]  Generating visualisation...")
    save_path = os.path.join(cfg.OUTPUTS_DIR, "phase1_visualisation.png")
    visualise_samples(
        gt_files  = gt_files,
        lr_files  = lr_tr_files,
        n_pairs   = 5,
        save_path = save_path,
    )

    print("\n" + "=" * 70)
    print("  Phase 1 COMPLETE. Ready to proceed to Phase 2: Data Preprocessing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
