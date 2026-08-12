"""
compare_models.py
=================
Dual-Model Benchmark & Comparison Tool.

Supports two evaluation modes:
  Mode 1 (Option 2 — Laptop Image Output Mode):
      Compare folder of Model 1 (Your Model) restored outputs vs
      folder of Model 2 (Friend's Model) restored outputs against Ground Truth.
      
      Usage:
      python compare_models.py \
          --gt dataset/train/GT \
          --m1 outputs/your_model_results \
          --m2 outputs/friend_model_results

  Mode 2 (Option 1 — Direct Checkpoint Mode):
      Run inference with two model weight files on test dataset and compare.

      Usage:
      python compare_models.py \
          --input dataset/NoisyLR \
          --gt dataset/train/GT \
          --w1 weights/amsr_net_best.pth \
          --w2 weights/amsr_net_epoch_0049.pth
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg
from dataset.semiconductor_dataset import normalise_lr, denormalise
from metrics.psnr import PSNRMetric
from metrics.ssim_metric import SSIMMetric
from models.amsr_net import AMSRNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"}


def load_image(path: Path) -> np.ndarray:
    """Load image as float32 array in [0, 1]."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        pil = Image.open(path).convert("L")
        arr = np.array(pil, dtype=np.float32) / 255.0

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {arr.shape} in {path}")
    return np.clip(arr, 0.0, 1.0)


def compute_metrics(
    restored: np.ndarray,
    gt: np.ndarray,
    psnr_fn: PSNRMetric,
    ssim_fn: SSIMMetric,
) -> Dict[str, float]:
    """Compute PSNR, SSIM, MSE, MAE, Edge Preservation Score."""
    pred_t = torch.from_numpy(restored).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

    # Resize GT if shape differs
    if pred_t.shape != gt_t.shape:
        gt_t = F.interpolate(gt_t, size=pred_t.shape[-2:], mode="bilinear", align_corners=False)

    gt_arr = gt_t[0, 0].numpy()

    with torch.no_grad():
        psnr_val = psnr_fn(pred_t, gt_t).item()
        ssim_val = ssim_fn(pred_t, gt_t).item()

    mse_val = float(np.mean((restored - gt_arr) ** 2))
    mae_val = float(np.mean(np.abs(restored - gt_arr)))

    # Sobel Edge Preservation metric
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    e_pred = torch.sqrt(F.conv2d(pred_t, kx, padding=1)**2 + F.conv2d(pred_t, ky, padding=1)**2)
    e_gt   = torch.sqrt(F.conv2d(gt_t, kx, padding=1)**2 + F.conv2d(gt_t, ky, padding=1)**2)
    edge_mae = float(torch.mean(torch.abs(e_pred - e_gt)).item())

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "mse": mse_val,
        "mae": mae_val,
        "edge_mae": edge_mae,
    }


def compare_image_directories(
    gt_dir: Path,
    m1_dir: Path,
    m2_dir: Path,
    output_report: Optional[Path] = None,
) -> Dict:
    """
    Compare restored output images from Model 1 and Model 2 against Ground Truth.
    """
    psnr_fn = PSNRMetric(data_range=1.0)
    ssim_fn = SSIMMetric(data_range=1.0, channels=1)

    m1_files = {p.stem: p for p in m1_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS}
    m2_files = {p.stem: p for p in m2_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS}
    gt_files = {p.stem: p for p in gt_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS}

    common_keys = sorted(set(gt_files.keys()).intersection(set(m1_files.keys())).intersection(set(m2_files.keys())))

    if not common_keys:
        logger.error("No matching image filenames found across GT, Model 1, and Model 2 directories!")
        return {}

    m1_results = []
    m2_results = []

    print("\n" + "=" * 80)
    print(f" ⚔️ DUAL-MODEL COMPARISON ARENA  |  {len(common_keys)} Common Test Images")
    print("=" * 80)
    print(f"{'Filename':<25s} | {'M1 PSNR':<9s} {'M1 SSIM':<9s} | {'M2 PSNR':<9s} {'M2 SSIM':<9s} | {'Winner'}")
    print("-" * 80)

    m1_wins = 0
    m2_wins = 0
    ties = 0

    for key in common_keys:
        gt_arr = load_image(gt_files[key])
        m1_arr = load_image(m1_files[key])
        m2_arr = load_image(m2_files[key])

        m1_met = compute_metrics(m1_arr, gt_arr, psnr_fn, ssim_fn)
        m2_met = compute_metrics(m2_arr, gt_arr, psnr_fn, ssim_fn)

        m1_results.append(m1_met)
        m2_results.append(m2_met)

        # Winner on this image (PSNR tie breaker SSIM)
        if m1_met["psnr"] > m2_met["psnr"] + 0.05:
            winner = "Your Model (M1) 🏆"
            m1_wins += 1
        elif m2_met["psnr"] > m1_met["psnr"] + 0.05:
            winner = "Friend's Model (M2) 🏆"
            m2_wins += 1
        else:
            if m1_met["ssim"] > m2_met["ssim"]:
                winner = "Your Model (M1) 🏆"
                m1_wins += 1
            elif m2_met["ssim"] > m1_met["ssim"]:
                winner = "Friend's Model (M2) 🏆"
                m2_wins += 1
            else:
                winner = "Tie 🤝"
                ties += 1

        print(
            f"{key[:24]:<25s} | "
            f"{m1_met['psnr']:7.2f}dB  {m1_met['ssim']:7.4f}  | "
            f"{m2_met['psnr']:7.2f}dB  {m2_met['ssim']:7.4f}  | "
            f"{winner}"
        )

    # Calculate dataset averages
    avg_m1 = {k: float(np.mean([r[k] for r in m1_results])) for k in m1_results[0]}
    avg_m2 = {k: float(np.mean([r[k] for r in m2_results])) for k in m2_results[0]}

    print("=" * 80)
    print(" 📊 OVERALL BENCHMARK SUMMARY")
    print("-" * 80)
    print(f" Metric             | Your Model (M1)        | Friend's Model (M2)    | Advantage")
    print("-" * 80)

    for metric, name in [
        ("psnr", "PSNR (dB) ↑"),
        ("ssim", "SSIM ↑"),
        ("mse", "MSE ↓"),
        ("mae", "MAE ↓"),
        ("edge_mae", "Edge Error ↓"),
    ]:
        v1 = avg_m1[metric]
        v2 = avg_m2[metric]
        if metric in ("psnr", "ssim"):
            diff = v1 - v2
            adv = f"M1 +{diff:.4f}" if diff > 0 else f"M2 +{-diff:.4f}"
        else:
            diff = v2 - v1
            adv = f"M1 (lower by {diff:.4f})" if diff > 0 else f"M2 (lower by {-diff:.4f})"

        fmt = "{:10.4f}" if metric != "psnr" else "{:10.2f} dB"
        print(f" {name:<18s} | {fmt.format(v1):<22s} | {fmt.format(v2):<22s} | {adv}")

    print("-" * 80)

    # Determine overall winner
    psnr_diff = avg_m1["psnr"] - avg_m2["psnr"]
    ssim_diff = avg_m1["ssim"] - avg_m2["ssim"]

    if psnr_diff > 0.1 and ssim_diff > 0.005:
        overall_winner = "YOUR MODEL (Model 1)"
        verdict = f"Your model achieved +{psnr_diff:.2f} dB higher PSNR and +{ssim_diff:.4f} higher SSIM!"
    elif psnr_diff < -0.1 and ssim_diff < -0.005:
        overall_winner = "FRIEND'S MODEL (Model 2)"
        verdict = f"Friend's model achieved +{-psnr_diff:.2f} dB higher PSNR and +{-ssim_diff:.4f} higher SSIM!"
    else:
        overall_winner = "MATCH / CLOSE PERFORMANCE"
        verdict = f"Both models perform comparably (PSNR diff: {psnr_diff:+.2f} dB)."

    print(f" 🏆 OVERALL WINNER: {overall_winner}")
    print(f" 📝 VERDICT: {verdict}")
    print("=" * 80 + "\n")

    summary_data = {
        "num_images": len(common_keys),
        "model1_wins": m1_wins,
        "model2_wins": m2_wins,
        "ties": ties,
        "model1_averages": avg_m1,
        "model2_averages": avg_m2,
        "overall_winner": overall_winner,
        "verdict": verdict,
    }

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(summary_data, f, indent=2)
        logger.info("Saved JSON report to %s", output_report)

    return summary_data


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Model 1 vs Model 2 Outputs against Ground Truth")
    parser.add_argument("--gt", required=True, type=Path, help="Directory containing Ground Truth clean images")
    parser.add_argument("--m1", required=True, type=Path, help="Directory containing Model 1 (Your Model) outputs")
    parser.add_argument("--m2", required=True, type=Path, help="Directory containing Model 2 (Friend's Model) outputs")
    parser.add_argument("--out", type=Path, default=Path("outputs/model_comparison_report.json"), help="Output JSON report path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.gt.exists():
        logger.error("GT directory not found: %s", args.gt)
        return 1
    if not args.m1.exists():
        logger.error("Model 1 output directory not found: %s", args.m1)
        return 1
    if not args.m2.exists():
        logger.error("Model 2 output directory not found: %s", args.m2)
        return 1

    compare_image_directories(args.gt, args.m1, args.m2, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
