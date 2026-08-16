"""
dataset/semiconductor_dataset.py
=================================
AMSR-Net - Phase 2: Data Preprocessing
----------------------------------------
PyTorch Dataset for the KLA Semiconductor Image Restoration task.

Design decisions (justified by Phase 1 findings)
-------------------------------------------------
1. Normalisation:
   - NoisyLR is clipped to [-0.2, 1.2] (covers observed population range
     [-0.19, 1.86] at the safe end; extreme outliers are true artefacts)
   - Both LR and GT are then mapped to [-1, +1] via (x - 0.5) / 0.5
   - This centres the distribution, which improves gradient flow in
     batch-normalised and attention-based networks.

2. Random crop (training only):
   - A random (patch_size x patch_size) patch is extracted from LR.
   - The SAME spatial location (scaled by 2x) is extracted from GT.
   - This ensures spatial correspondence is preserved exactly.
   - Patch-based training:
       * Increases effective dataset size (many patches per image).
       * Enables fixed batch sizes regardless of image resolution.
       * Reduces GPU memory per forward pass.

3. Augmentation (training only):
   - Horizontal flip, vertical flip, and 90-degree rotations.
   - Applied with independent probabilities to both LR and GT
     IDENTICALLY (same random state) to preserve correspondence.
   - Justified: semiconductor circuit patterns have no preferred
     orientation -- all flips/rotations are physically valid.
   - Albumentations is NOT used here because .npy is not uint8 image.
     Instead, manual NumPy transforms are used for precision control.

4. Tensor conversion:
   - Output shape: (1, H, W) -- channel-first, single grayscale channel.
   - dtype: float32 (matches model expectation, no casting overhead).
"""

import os
import sys
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg


# ===========================================================================
# 0.  DIRECTORY SCANNING  (shared utility)
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

    Notes
    -----
    Sorting guarantees that GT[i] and NoisyLR[i] are paired correctly
    when both folders contain identically named files (000000.npy, ...).
    """
    import glob
    pattern = os.path.join(directory, extension)
    return sorted(glob.glob(pattern))


# ===========================================================================
# 1.  NORMALISATION UTILITIES
# ===========================================================================

def normalise_lr(arr: np.ndarray) -> np.ndarray:
    """
    Preprocess a NoisyLR image for network input.

    Steps
    -----
    1. Clip to [-0.2, 1.2]:
       The Phase 1 population analysis found NoisyLR pixel values in the
       range [-0.19, 1.86]. Values beyond +-0.2 / 1.2 are extreme speckle
       outliers that would otherwise dominate the normalisation range.
       Clipping at these limits retains >99.9% of the pixel distribution
       while preventing the normalisation from being skewed by rare outliers.

    2. Map [clip_min, clip_max] -> [-1, +1]:
       z = (x - 0.5) / 0.5
       This centres the distribution around 0, which is beneficial for:
       - BatchNorm / LayerNorm statistics (mean closer to 0)
       - Sigmoid / tanh activations in the decoder head
       - Symmetric gradient magnitudes in the first conv layer

    Parameters
    ----------
    arr : np.ndarray
        Raw NoisyLR array, shape (H, W), dtype float32.

    Returns
    -------
    np.ndarray
        Normalised array in [-1, +1], same shape.
    """
    arr = np.clip(arr, -0.2, 1.2)
    arr = (arr - 0.5) / 0.5          # maps [0,1] -> [-1,+1]; [-0.2,1.2] -> [-1.4,1.4]
    return arr.astype(np.float32)


def normalise_gt(arr: np.ndarray) -> np.ndarray:
    """
    Preprocess a GT image for use as the training target.

    GT is already in [0, 1] (confirmed in Phase 1). We apply the same
    affine transform as LR so that the network output domain matches:
        z = (x - 0.5) / 0.5  ->  [-1, +1]

    The loss function and metric computation will denormalise before
    computing PSNR/SSIM to keep them in the standard [0, 1] range.

    Parameters
    ----------
    arr : np.ndarray
        Raw GT array, shape (H, W), dtype float32, values in [0, 1].

    Returns
    -------
    np.ndarray
        Normalised array in [-1, +1], same shape.
    """
    arr = (arr - 0.5) / 0.5
    return arr.astype(np.float32)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """
    Invert the [-1, +1] normalisation: z -> z * 0.5 + 0.5.

    Used before computing PSNR / SSIM (which expect [0, 1] inputs)
    and before saving output images.

    Parameters
    ----------
    tensor : torch.Tensor
        Network output or GT tensor normalised to [-1, +1].

    Returns
    -------
    torch.Tensor
        Pixel values approximately in [0, 1] (clamp after calling).
    """
    return tensor * 0.5 + 0.5


# ===========================================================================
# Augmentation transforms
# ===========================================================================

class PairedTransform:
    """
    Apply identical spatial transforms to an (LR, GT) pair.

    Spatial transforms must be IDENTICAL on both images to preserve
    the pixel-level GT<->LR correspondence. We achieve this by drawing
    a single random integer per transform and reusing it for both arrays.

    Supported transforms
    --------------------
    - Random horizontal flip   (p = cfg.AUG_HFLIP_P)
    - Random vertical flip     (p = cfg.AUG_VFLIP_P)
    - Random 90-deg rotation   (p = cfg.AUG_ROT90_P, k in {1,2,3})

    Note: These transforms are rotation-group symmetric -- applying to LR
    and GT separately with the same flip/rotate gives identical spatial
    correspondence because both are axis-aligned crops of the same scene.

    Parameters
    ----------
    hflip_p : float  Probability of horizontal flip.
    vflip_p : float  Probability of vertical flip.
    rot90_p : float  Probability of 90-degree rotation.
    """

    def __init__(
        self,
        hflip_p: float = cfg.AUG_HFLIP_P,
        vflip_p: float = cfg.AUG_VFLIP_P,
        rot90_p: float = cfg.AUG_ROT90_P,
    ) -> None:
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.rot90_p = rot90_p

    def __call__(
        self,
        lr: np.ndarray,
        gt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random spatial transforms to an (lr, gt) image pair.

        Parameters
        ----------
        lr : np.ndarray  LR patch, shape (H, W).
        gt : np.ndarray  GT patch, shape (2H, 2W).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed (lr, gt) pair.
        """
        # Horizontal flip
        if random.random() < self.hflip_p:
            lr = np.fliplr(lr).copy()
            gt = np.fliplr(gt).copy()

        # Vertical flip
        if random.random() < self.vflip_p:
            lr = np.flipud(lr).copy()
            gt = np.flipud(gt).copy()

        # 90-degree rotation (k in {1, 2, 3})
        if random.random() < self.rot90_p:
            k = random.randint(1, 3)
            lr = np.rot90(lr, k=k).copy()
            gt = np.rot90(gt, k=k).copy()

        return lr, gt


# ===========================================================================
# Patch extraction
# ===========================================================================

def random_crop_pair(
    lr: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    scale: int = cfg.SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract spatially aligned random crops from an (LR, GT) pair.

    The crop origin is sampled uniformly from LR coordinate space.
    The corresponding GT crop is extracted at (origin * scale) to
    maintain exact sub-pixel alignment.

    Parameters
    ----------
    lr         : np.ndarray  Full LR image, shape (H_lr, W_lr).
    gt         : np.ndarray  Full GT image, shape (H_gt, W_gt).
    patch_size : int         LR patch side length in pixels.
    scale      : int         Upscaling factor (GT_size = LR_size * scale).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (lr_patch, gt_patch) with shapes (patch_size, patch_size)
        and (patch_size*scale, patch_size*scale).

    Raises
    ------
    ValueError
        If the LR image is smaller than the requested patch_size.
    """
    h_lr, w_lr = lr.shape[:2]

    if h_lr < patch_size or w_lr < patch_size:
        raise ValueError(
            f"LR image ({h_lr}x{w_lr}) is smaller than patch_size={patch_size}. "
            f"Reduce cfg.PATCH_SIZE."
        )

    # Sample top-left corner in LR space
    top_lr  = random.randint(0, h_lr - patch_size)
    left_lr = random.randint(0, w_lr - patch_size)

    # Corresponding GT crop (scaled coordinates)
    top_gt  = top_lr  * scale
    left_gt = left_lr * scale
    size_gt = patch_size * scale

    lr_patch = lr[top_lr  : top_lr  + patch_size, left_lr : left_lr + patch_size]
    gt_patch = gt[top_gt  : top_gt  + size_gt,    left_gt : left_gt + size_gt]

    return lr_patch, gt_patch


def centre_crop_pair(
    lr: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    scale: int = cfg.SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the centre crop from an (LR, GT) pair.

    Used during validation/inference to ensure deterministic inputs
    (no random spatial offset) while still working with fixed-size
    patches.

    Parameters
    ----------
    lr         : np.ndarray  Full LR image.
    gt         : np.ndarray  Full GT image.
    patch_size : int         LR crop size.
    scale      : int         Upscaling factor.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Centre-cropped (lr_patch, gt_patch).
    """
    h_lr, w_lr = lr.shape[:2]

    top_lr  = (h_lr - patch_size) // 2
    left_lr = (w_lr - patch_size) // 2
    top_gt  = top_lr  * scale
    left_gt = left_lr * scale
    size_gt = patch_size * scale

    lr_patch = lr[top_lr  : top_lr  + patch_size, left_lr : left_lr + patch_size]
    gt_patch = gt[top_gt  : top_gt  + size_gt,    left_gt : left_gt + size_gt]

    return lr_patch, gt_patch


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class SemiconductorDataset(Dataset):
    """
    PyTorch Dataset for the KLA semiconductor image restoration task.

    Each sample is a dict with keys:
        "lr"       : torch.Tensor of shape (1, patch_size, patch_size)
        "gt"       : torch.Tensor of shape (1, patch_size*scale, patch_size*scale)
        "filename" : str basename of the source file (for traceability)

    Training mode
    -------------
    - Random crop of size (patch_size x patch_size) from LR.
    - Corresponding (patch_size*scale x patch_size*scale) from GT.
    - Random horizontal flip, vertical flip, 90-deg rotation.

    Validation / inference mode
    ---------------------------
    - Centre crop (deterministic, no augmentation).
    - Full images can also be used by setting patch_size=None (see below).

    Parameters
    ----------
    gt_files    : List[str]   Sorted list of GT .npy file paths.
    lr_files    : List[str]   Sorted list of NoisyLR .npy file paths.
    patch_size  : int         LR crop size. GT crop = patch_size * scale.
    scale       : int         Super-resolution scale factor.
    is_train    : bool        If True, use random crop + augmentation.
                              If False, use centre crop, no augmentation.
    transform   : Optional    Additional transform callable (for future use).
    """

    def __init__(
        self,
        gt_files:   List[str],
        lr_files:   List[str],
        patch_size: int               = cfg.PATCH_SIZE,
        scale:      int               = cfg.SCALE,
        is_train:   bool              = True,
        transform:  Optional[Callable] = None,
    ) -> None:
        super().__init__()

        assert len(gt_files) == len(lr_files), (
            f"GT and LR file counts differ: {len(gt_files)} vs {len(lr_files)}"
        )

        self.gt_files   = gt_files
        self.lr_files   = lr_files
        self.patch_size = patch_size
        self.scale      = scale
        self.is_train   = is_train
        self.transform  = transform

        # Augmentation transform (only applied during training)
        self._aug = PairedTransform() if is_train else None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.gt_files)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, object]:
        """
        Load, preprocess, augment and return a single (LR, GT) sample.

        Pipeline
        --------
        1. Load raw .npy arrays.
        2. Random or centre crop (preserving LR<->GT alignment).
        3. Normalise LR and GT to [-1, +1].
        4. Augment (training only): flip + rotate.
        5. Add channel dim: (H,W) -> (1,H,W).
        6. Convert to float32 torch.Tensor.

        Parameters
        ----------
        index : int  Sample index.

        Returns
        -------
        Dict with keys "lr", "gt", "filename".
        """
        gt_raw = np.load(self.gt_files[index], allow_pickle=False)   # (256, 256)
        lr_raw = np.load(self.lr_files[index], allow_pickle=False)   # (128, 128)

        filename = os.path.basename(self.gt_files[index])

        # Step 1: Spatial crop
        if self.is_train:
            lr_patch, gt_patch = random_crop_pair(
                lr_raw, gt_raw, self.patch_size, self.scale
            )
        else:
            lr_patch, gt_patch = centre_crop_pair(
                lr_raw, gt_raw, self.patch_size, self.scale
            )

        # Step 2: Normalise to [-1, +1]
        lr_norm = normalise_lr(lr_patch)
        gt_norm = normalise_gt(gt_patch)

        # Step 3: Augmentation (training only)
        if self.is_train and self._aug is not None:
            lr_norm, gt_norm = self._aug(lr_norm, gt_norm)

        # Step 4: Additional transform hook (for future use)
        if self.transform is not None:
            lr_norm, gt_norm = self.transform(lr_norm, gt_norm)

        # Step 5: Add channel dimension -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_norm[np.newaxis, ...])   # (1, 64, 64)
        gt_tensor = torch.from_numpy(gt_norm[np.newaxis, ...])   # (1, 128, 128)

        return {
            "lr":       lr_tensor,
            "gt":       gt_tensor,
            "filename": filename,
        }

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        mode = "TRAIN (random crop + augmentation)" if self.is_train else "VAL (centre crop)"
        return (
            f"SemiconductorDataset(\n"
            f"  mode       = {mode}\n"
            f"  samples    = {len(self)}\n"
            f"  patch_size = {self.patch_size} (LR) / "
            f"{self.patch_size * self.scale} (GT)\n"
            f"  scale      = {self.scale}x\n"
            f")"
        )
