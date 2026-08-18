"""
run.py
======
Silicon Crafters — KLA Problem Statement
AI-Based Restoration of Degraded Semiconductor Images

Entry Point
-----------
    python run.py <input-dir> <output-dir>

Behaviour
---------
* Reads every .npy file from <input-dir> (non-recursive first pass;
  falls back to recursive if the top level is empty).
* Creates <output-dir> if it does not already exist.
* Runs AMSR-Net inference (2× Denoising + Super-Resolution) on each image.
* Saves one restored .npy file per input, with the SAME filename.
* Output arrays: float32, shape (H, W), values strictly in [0, 1].
* Runs on NVIDIA GPU automatically; falls back to CPU if unavailable.
* Fully offline — no internet, no API keys, no additional downloads.

Model
-----
AMSR-Net: Adaptive Multi-Expert Semiconductor Restoration Network
Architecture  : CNN ResBlocks + Restormer MDTA + Swin Transformer Pairs
Task          : Joint denoising + 2× super-resolution (128×128 → 256×256)
Weights       : models/amsr_net_best.pth  (~12 MB, included in submission)
"""

import sys
import os
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add models/ to sys.path so all sub-modules are importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

import numpy as np
import torch

from amsr_net import AMSRNet
from dataset.semiconductor_dataset import normalise_lr, denormalise
import config as cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WEIGHTS_PATH = _MODELS_DIR / "amsr_net_best.pth"


# ===========================================================================
# MODEL
# ===========================================================================

def load_model(device: torch.device) -> AMSRNet:
    """
    Instantiate AMSRNet and load the best checkpoint.
    All hyper-parameters are read from models/config.py.
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

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}\n"
            "Ensure 'models/amsr_net_best.pth' is present in the submission folder."
        )

    checkpoint = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info("Model loaded from: %s  |  Device: %s", WEIGHTS_PATH.name, device)
    return model


# ===========================================================================
# IMAGE I/O
# ===========================================================================

def find_npy_files(directory: Path) -> list:
    """
    Return a sorted list of all .npy files in the given directory.
    Searches non-recursively first; falls back to recursive if none found.
    """
    files = sorted(directory.glob("*.npy"))
    if not files:
        # Recursive fallback (handles nested sub-directories)
        files = sorted(directory.rglob("*.npy"))
    return files


def load_npy(path: Path) -> np.ndarray:
    """
    Load a .npy file and return a float32 2-D (H, W) array.

    Handles:
    - (H, W)    — used directly
    - (H, W, 1) — squeezed to (H, W)
    - (H, W, C) — averaged across channels to grayscale
    """
    arr = np.load(path, allow_pickle=False).astype(np.float32)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]           # (H, W, 1) → (H, W)
        else:
            arr = arr.mean(axis=-1)      # (H, W, C) → (H, W) grayscale mean
    if arr.ndim != 2:
        raise ValueError(f"Unexpected array shape {arr.shape} in {path.name}")
    if arr.size == 0:
        raise ValueError(f"Empty array in {path.name}")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError(f"NaN/Inf values detected in {path.name} — skipping")

    return arr


def save_npy(arr: np.ndarray, out_path: Path) -> None:
    """
    Validate and save a float32 (H, W) array as a .npy file.

    Guarantees:
    - Values clipped to [0, 1]
    - No NaN or Inf
    - dtype float32
    """
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)

    # Replace any residual NaN/Inf with 0.0 (safety net)
    arr = np.where(np.isfinite(arr), arr, 0.0)

    assert arr.ndim == 2, f"Output must be 2-D (H, W), got shape {arr.shape}"
    assert arr.min() >= 0.0 and arr.max() <= 1.0, "Output values out of [0, 1]"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


# ===========================================================================
# INFERENCE
# ===========================================================================

def preprocess(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Apply training-identical normalisation and move to device.

    Pipeline (matches semiconductor_dataset.normalise_lr):
        clip(-0.2, 1.2) → (x - 0.5) / 0.5 → tensor (1, 1, H, W)
    """
    normed = normalise_lr(arr)                                    # (H, W) float32
    tensor = torch.from_numpy(normed[np.newaxis, np.newaxis, :]) # (1, 1, H, W)
    return tensor.to(device)


def postprocess(out_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalise network output to [0, 1] float32 numpy array (H, W).
    Matches semiconductor_dataset.denormalise: z * 0.5 + 0.5.
    """
    out_01 = denormalise(out_tensor[0, 0]).clamp(0.0, 1.0)
    return out_01.float().cpu().numpy()


@torch.inference_mode()
def restore_image(
    model: AMSRNet,
    arr: np.ndarray,
    device: torch.device,
) -> tuple:
    """
    Run a single image through AMSR-Net.

    Returns
    -------
    restored : np.ndarray  (H*2, W*2) float32 in [0, 1]
    ms       : float       inference latency in milliseconds
    """
    tensor = preprocess(arr, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        out = model(tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return postprocess(out), (t1 - t0) * 1000.0


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> int:
    # -----------------------------------------------------------------------
    # Parse positional arguments
    # -----------------------------------------------------------------------
    if len(sys.argv) < 3:
        print(
            "Usage:  python run.py <input-dir> <output-dir>\n"
            "\n"
            "  input-dir   Directory containing degraded .npy images (NoisyLR).\n"
            "  output-dir  Directory where restored .npy images will be saved.\n"
            "\n"
            "Example:\n"
            "  python run.py ./dataset/NoisyLR ./outputs/restored\n",
            file=sys.stderr,
        )
        return 1

    input_dir  = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    # -----------------------------------------------------------------------
    # Validate input directory
    # -----------------------------------------------------------------------
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1
    if not input_dir.is_dir():
        logger.error("Input path is not a directory: %s", input_dir)
        return 1

    # -----------------------------------------------------------------------
    # Device selection (NVIDIA GPU preferred)
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("GPU detected: %s", gpu_name)
    else:
        logger.info("No GPU detected — running on CPU.")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    t_start = time.perf_counter()
    try:
        model = load_model(device)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    load_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info("Model ready in %.0f ms", load_ms)

    # -----------------------------------------------------------------------
    # Discover input files
    # -----------------------------------------------------------------------
    input_files = find_npy_files(input_dir)
    if not input_files:
        logger.error("No .npy files found in: %s", input_dir)
        return 1
    logger.info("Found %d .npy file(s) to process.", len(input_files))

    # -----------------------------------------------------------------------
    # Create output directory
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Batch inference
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print(f"  Silicon Crafters -- AMSR-Net Restoration")
    print(f"  Images : {len(input_files)}  |  Device : {device}")
    print("=" * 68)

    n_ok     = 0
    n_fail   = 0
    total_ms = 0.0

    for idx, lr_path in enumerate(input_files, 1):
        try:
            # Load
            raw = load_npy(lr_path)

            # Infer
            restored, inf_ms = restore_image(model, raw, device)
            total_ms += inf_ms

            # Save — same filename, .npy extension, flat in output_dir
            out_path = output_dir / lr_path.name
            save_npy(restored, out_path)

            print(f"  [{idx:4d}/{len(input_files)}]  {lr_path.name:<28s}  "
                  f"{raw.shape[0]}x{raw.shape[1]} -> {restored.shape[0]}x{restored.shape[1]}  "
                  f"{inf_ms:6.1f} ms")
            n_ok += 1

        except Exception as exc:
            logger.warning("SKIPPED [%s]: %s", lr_path.name, exc)
            n_fail += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 68)
    print(f"  Done.  Success: {n_ok}  |  Failed: {n_fail}")
    if n_ok > 0:
        avg_ms = total_ms / n_ok
        print(f"  Avg inference : {avg_ms:.1f} ms/image  ({1000/avg_ms:.1f} img/s)")
    print(f"  Outputs saved : {output_dir}")
    print("=" * 68 + "\n")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
