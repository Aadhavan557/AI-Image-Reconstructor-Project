"""
evaluation.py
=============
Phase 16 — KLA-Compatible Standalone Evaluation Script
--------------------------------------------------------
Loads the trained AMSR-Net model and restores all images in a given
input directory, saving outputs to the specified output directory.

Usage
-----
    python evaluation.py --input <INPUT_DIR> --output <OUTPUT_DIR>

Optional flags
--------------
    --gt      <GT_DIR>       Ground-truth directory for PSNR/SSIM calculation.
    --weights <CKPT_PATH>    Override default weights path.
    --device  cpu|cuda       Force device (default: auto).
    --ext     npy|png|...    Input file extension filter (default: npy).

Example
-------
    python evaluation.py \\
        --input  ../dataset/NoisyLR \\
        --output ./outputs/eval_results \\
        --gt     ../dataset/train/GT
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of CWD
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as cfg
from models.amsr_net import AMSRNet
from dataset.semiconductor_dataset import normalise_lr, denormalise
from metrics.psnr import PSNRMetric
from metrics.ssim_metric import SSIMMetric

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# MODEL UTILITIES
# ===========================================================================

def build_model(weights_path: Path, device: torch.device) -> AMSRNet:
    """
    Instantiate AMSRNet with config defaults and load checkpoint.

    Parameters
    ----------
    weights_path : Path   Path to .pth checkpoint file.
    device       : torch.device

    Returns
    -------
    AMSRNet  in eval() mode on the target device.
    """
    model = AMSRNet(
        in_channels      = cfg.CHANNELS,
        dim              = cfg.AMSRNET_DIM,
        encoder_blocks   = cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks = cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks      = cfg.AMSRNET_SWIN_BLOCKS,
        window_size      = cfg.AMSRNET_WINDOW_SIZE,
        num_heads        = cfg.AMSRNET_NUM_HEADS,
    )

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info("Loaded model from %s on %s", weights_path, device)
    return model


# ===========================================================================
# IMAGE I/O
# ===========================================================================

SUPPORTED_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def find_input_files(input_dir: Path, ext_filter: Optional[str] = None) -> List[Path]:
    """
    Recursively find all supported image files in input_dir.

    Parameters
    ----------
    input_dir  : Path
    ext_filter : str | None   e.g. '.npy'. None = all supported types.
    """
    files: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if ext_filter:
            if suffix == ext_filter.lower():
                files.append(path)
        elif suffix in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def load_image(path: Path) -> np.ndarray:
    """
    Load an image file as a float32 (H, W) numpy array in approximately [0, 1].

    Supports .npy (loaded directly) and image formats (converted to grayscale).

    Raises
    ------
    ValueError  : for unsupported formats, wrong dims, NaN/Inf values.
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape} in {path}")
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif"}:
        pil = Image.open(path).convert("L")
        arr = np.array(pil, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"NaN/Inf values detected in: {path}")

    return arr


def save_image(arr: np.ndarray, out_path: Path) -> None:
    """
    Save a float32 [0, 1] (H, W) array as uint8 PNG.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(uint8, mode="L").save(out_path)


# ===========================================================================
# PREPROCESSING / POSTPROCESSING
# ===========================================================================

def preprocess(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Apply training-identical normalisation and move to device.

    Pipeline (matches semiconductor_dataset.normalise_lr exactly):
        clip(-0.2, 1.2) → (x - 0.5) / 0.5 → float32 tensor (1, 1, H, W)
    """
    norm = normalise_lr(arr)                            # (H, W) float32
    tensor = torch.from_numpy(norm[np.newaxis, np.newaxis, ...])  # (1,1,H,W)
    return tensor.to(device)


def postprocess(out_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalise network output to [0, 1] float32 numpy array (H, W).
    Matches semiconductor_dataset.denormalise: z * 0.5 + 0.5.
    """
    out_01 = denormalise(out_tensor[0, 0]).clamp(0.0, 1.0)
    return out_01.float().cpu().numpy()


# ===========================================================================
# INFERENCE
# ===========================================================================

def run_single(
    model: AMSRNet,
    arr: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, float]:
    """
    Run AMSR-Net inference on a single image.

    Returns
    -------
    restored_arr : np.ndarray (H*2, W*2) float32 in [0, 1]
    inference_ms : float      pure model inference time (ms)
    """
    tensor = preprocess(arr, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return postprocess(out), (t1 - t0) * 1000.0


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(
    restored: np.ndarray,
    gt: np.ndarray,
    psnr_fn: PSNRMetric,
    ssim_fn: SSIMMetric,
) -> dict:
    """
    Compute PSNR, SSIM, MSE, MAE. Both arrays must be float32 in [0, 1].
    """
    pred_t = torch.from_numpy(restored).unsqueeze(0).unsqueeze(0)
    gt_t   = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

    # Resize GT to match restored if shape differs
    if pred_t.shape != gt_t.shape:
        import torch.nn.functional as F
        gt_t = F.interpolate(gt_t, size=pred_t.shape[-2:], mode="bilinear", align_corners=False)

    with torch.no_grad():
        psnr_val = psnr_fn(pred_t, gt_t).item()
        ssim_val = ssim_fn(pred_t, gt_t).item()

    mse = float(np.mean((restored - gt_t[0, 0].numpy()) ** 2))
    mae = float(np.mean(np.abs(restored - gt_t[0, 0].numpy())))

    return {"psnr": psnr_val, "ssim": ssim_val, "mse": mse, "mae": mae}


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AMSR-Net Evaluation Script — Phase 16"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Directory containing degraded NoisyLR images.",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Directory to save restored images.",
    )
    parser.add_argument(
        "--gt", type=Path, default=None,
        help="(Optional) Ground-truth directory. Enables PSNR/SSIM calculation.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path(cfg.BEST_MODEL_PATH),
        help=f"Path to model checkpoint (default: {cfg.BEST_MODEL_PATH}).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto).",
    )
    parser.add_argument(
        "--ext", type=str, default=None,
        help="Filter input files by extension, e.g. '.npy' (default: all supported).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --- Device selection ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    logger.info("Device: %s (%s)", device, gpu_name)

    # --- Load model ---
    t_load_start = time.perf_counter()
    try:
        model = build_model(args.weights, device)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    t_load_end = time.perf_counter()
    load_ms = (t_load_end - t_load_start) * 1000.0
    logger.info("Model loaded in %.1f ms", load_ms)

    # --- Find input files ---
    input_dir: Path = args.input
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    input_files = find_input_files(input_dir, args.ext)
    if not input_files:
        logger.error("No supported images found in %s", input_dir)
        return 1

    logger.info("Found %d images to process.", len(input_files))

    # --- Output directory ---
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Metrics setup ---
    compute_gt_metrics = args.gt is not None and args.gt.exists()
    psnr_fn = PSNRMetric(data_range=1.0)
    ssim_fn = SSIMMetric(data_range=1.0, channels=1)

    # --- Process images ---
    total_inference_ms = 0.0
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    n_success = 0
    n_failed  = 0

    print("\n" + "=" * 70)
    print(f"  AMSR-Net Evaluation  |  {len(input_files)} images  |  {device}")
    print("=" * 70)

    for idx, lr_path in enumerate(input_files, 1):
        try:
            # Load input
            raw_arr = load_image(lr_path)

            # Inference
            restored_arr, inf_ms = run_single(model, raw_arr, device)
            total_inference_ms += inf_ms

            # Save output — preserve relative path structure
            rel = lr_path.relative_to(input_dir)
            out_path = args.output / rel.with_suffix(".png")
            save_image(restored_arr, out_path)

            # Optional GT metrics
            metric_str = ""
            if compute_gt_metrics:
                gt_path = args.gt / rel
                if gt_path.exists():
                    gt_arr = load_image(gt_path)
                    m = compute_metrics(restored_arr, gt_arr, psnr_fn, ssim_fn)
                    all_psnr.append(m["psnr"])
                    all_ssim.append(m["ssim"])
                    metric_str = f" | PSNR={m['psnr']:.2f}dB SSIM={m['ssim']:.4f}"

            print(f"  [{idx:4d}/{len(input_files)}] {lr_path.name:<30s} "
                  f"{inf_ms:6.1f}ms{metric_str}")
            n_success += 1

        except Exception as e:
            logger.warning("FAILED [%s]: %s", lr_path.name, e)
            n_failed += 1
            continue

    # --- Final summary ---
    print("\n" + "=" * 70)
    print(f"  [OK] Completed: {n_success} | [FAIL] Failed: {n_failed}")
    print(f"  Model load time  : {load_ms:.1f} ms")
    if n_success > 0:
        avg_ms = total_inference_ms / n_success
        fps    = 1000.0 / avg_ms
        print(f"  Avg inference    : {avg_ms:.1f} ms/image  ({fps:.1f} img/s)")
        print(f"  Total inference  : {total_inference_ms/1000:.2f} s")
    if all_psnr:
        print(f"  Mean PSNR        : {np.mean(all_psnr):.4f} dB")
        print(f"  Mean SSIM        : {np.mean(all_ssim):.4f}")
    print(f"  Outputs saved to : {args.output.resolve()}")
    print("=" * 70)

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
