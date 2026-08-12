"""
noise_generator.py
==================
Synthetic Noise Generation Suite for Image Restoration & Model Stress Testing.

Provides standard noise injection functions used in physical optics,
electron microscopy (SEM), and computer vision benchmarking:
  1. Additive Gaussian Noise (sensor thermal noise)
  2. Poisson / Shot Noise (quantum photon count noise)
  3. Salt & Pepper / Impulse Noise (defective detector pixels)
  4. Speckle / Multiplicative Noise (coherent beam interference)
  5. Mixed Real-World SEM Noise (blur + Poisson + Gaussian)

All input arrays must be float32 in range [0, 1].
All outputs are float32 clipped to [0, 1].
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard Noise Descriptions for UI & Docs
NOISE_DESCRIPTIONS = {
    "gaussian": {
        "name": "Additive Gaussian Noise",
        "description": "Simulates electronic sensor thermal noise (constant variance across pixel values).",
        "params": "std (sigma) controls noise amplitude."
    },
    "poisson": {
        "name": "Poisson (Shot) Noise",
        "description": "Simulates quantum photon count fluctuations common in electron microscopy (SEM/TEM).",
        "params": "scale controls photon count intensity (lower scale = higher noise)."
    },
    "salt_pepper": {
        "name": "Salt & Pepper (Impulse) Noise",
        "description": "Simulates dead pixels and digital bit corruption (random black and white pixels).",
        "params": "amount controls percentage of corrupted pixels."
    },
    "speckle": {
        "name": "Speckle (Multiplicative) Noise",
        "description": "Simulates coherent laser or electron beam interference patterns.",
        "params": "std controls multiplicative noise variance."
    },
    "mixed_sem": {
        "name": "Mixed Real-World SEM Noise",
        "description": "Realistic SEM acquisition model combining spatial low-pass blur, Poisson shot noise, and Gaussian thermal noise.",
        "params": "intensity scales overall degradation severity."
    }
}


def add_gaussian_noise(
    image: np.ndarray,
    std: float = 0.05,
    mean: float = 0.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add additive Gaussian noise: I_noisy = I + N(mean, std^2).

    Parameters
    ----------
    image : np.ndarray
        Float32 image array in range [0, 1].
    std : float
        Standard deviation (sigma) of Gaussian distribution.
    mean : float
        Mean of noise distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image array in range [0, 1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def add_poisson_noise(
    image: np.ndarray,
    scale: float = 25.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Poisson (shot) noise: I_noisy = Poisson(I * scale) / scale.
    Lower scale values yield higher noise variance.

    Parameters
    ----------
    image : np.ndarray
        Float32 image array in range [0, 1].
    scale : float
        Scaling factor representing peak photon count (e.g., 10.0 to 100.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image array in range [0, 1].
    """
    rng = np.random.default_rng(seed)
    img_clamped = np.clip(image.astype(np.float32), 1e-6, 1.0)
    # Scaled photon counts
    scaled = img_clamped * scale
    noisy_counts = rng.poisson(scaled).astype(np.float32)
    noisy = noisy_counts / scale
    return np.clip(noisy, 0.0, 1.0)


def add_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Salt and Pepper impulse noise.

    Parameters
    ----------
    image : np.ndarray
        Float32 image array in range [0, 1].
    amount : float
        Fraction of pixels to corrupt (e.g. 0.02 = 2%).
    salt_vs_pepper : float
        Ratio of salt (white=1.0) vs pepper (black=0.0) noise (default 0.5).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image array in range [0, 1].
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32).copy()
    num_noise = int(np.ceil(amount * image.size))

    # Choose random pixel coordinates
    coords = [rng.integers(0, i - 1, int(num_noise)) for i in image.shape]

    # Salt (White = 1.0)
    num_salt = int(np.ceil(num_noise * salt_vs_pepper))
    salt_coords = tuple(c[:num_salt] for c in coords)
    noisy[salt_coords] = 1.0

    # Pepper (Black = 0.0)
    pepper_coords = tuple(c[num_salt:] for c in coords)
    noisy[pepper_coords] = 0.0

    return np.clip(noisy, 0.0, 1.0)


def add_speckle_noise(
    image: np.ndarray,
    std: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Speckle (multiplicative) noise: I_noisy = I + I * N(0, std^2).

    Parameters
    ----------
    image : np.ndarray
        Float32 image array in range [0, 1].
    std : float
        Standard deviation of multiplicative noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image array in range [0, 1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=std, size=image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + image.astype(np.float32) * noise
    return np.clip(noisy, 0.0, 1.0)


def add_mixed_sem_noise(
    image: np.ndarray,
    intensity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate realistic SEM acquisition degradation:
    Blur -> Poisson Shot Noise -> Gaussian Thermal Noise.

    Parameters
    ----------
    image : np.ndarray
        Float32 image array in range [0, 1].
    intensity : float
        Overall severity multiplier (1.0 = standard, 2.0 = severe).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image array in range [0, 1].
    """
    # 1. Spatial Gaussian Blur to simulate beam defocus
    from scipy.ndimage import gaussian_filter
    blur_sigma = 0.5 * intensity
    blurred = gaussian_filter(image.astype(np.float32), sigma=blur_sigma)

    # 2. Poisson shot noise
    poisson_scale = max(5.0, 30.0 / intensity)
    poisson_noisy = add_poisson_noise(blurred, scale=poisson_scale, seed=seed)

    # 3. Additive Gaussian thermal noise
    gauss_std = 0.03 * intensity
    final_noisy = add_gaussian_noise(poisson_noisy, std=gauss_std, seed=seed)

    return final_noisy


def generate_noise(
    image: np.ndarray,
    noise_type: str = "gaussian",
    intensity: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Unified function to apply noise by name and intensity multiplier.

    Parameters
    ----------
    image : np.ndarray
        Float32 array in range [0, 1].
    noise_type : str
        One of 'gaussian', 'poisson', 'salt_pepper', 'speckle', 'mixed_sem'.
    intensity : float
        Severity multiplier (0.5 = mild, 1.0 = medium, 2.0 = severe).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image in [0, 1].
    """
    noise_type = noise_type.lower().strip()

    if noise_type == "gaussian":
        std = 0.05 * intensity
        return add_gaussian_noise(image, std=std, seed=seed)

    elif noise_type == "poisson":
        scale = max(2.0, 25.0 / intensity)
        return add_poisson_noise(image, scale=scale, seed=seed)

    elif noise_type == "salt_pepper":
        amount = min(0.2, 0.02 * intensity)
        return add_salt_pepper_noise(image, amount=amount, seed=seed)

    elif noise_type == "speckle":
        std = 0.1 * intensity
        return add_speckle_noise(image, std=std, seed=seed)

    elif noise_type in ("mixed_sem", "mixed"):
        return add_mixed_sem_noise(image, intensity=intensity, seed=seed)

    else:
        raise ValueError(
            f"Unknown noise_type: '{noise_type}'. "
            f"Supported types: {list(NOISE_DESCRIPTIONS.keys())}"
        )


def export_benchmark_suite(
    gt_dir: Union[str, Path],
    output_dir: Union[str, Path],
    noise_types: Optional[List[str]] = None,
    intensity: float = 1.0,
    seed: int = 42,
    max_images: Optional[int] = None
) -> Tuple[int, Path]:
    """
    Batch process Ground Truth images to create a standardized noisy test suite
    that can be shared with colleagues/friends for comparison testing.

    Parameters
    ----------
    gt_dir : str | Path
        Directory containing clean Ground Truth images.
    output_dir : str | Path
        Directory to save corrupted test images.
    noise_types : list of str, optional
        Noise types to generate. Default: ['gaussian', 'poisson', 'speckle', 'mixed_sem'].
    intensity : float
        Noise severity scale.
    seed : int
        Base random seed.
    max_images : int, optional
        Maximum number of GT images to process (None = all).

    Returns
    -------
    (count, output_dir_path)
    """
    gt_path = Path(gt_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if noise_types is None:
        noise_types = ["gaussian", "poisson", "speckle", "mixed_sem"]

    valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".npy"}
    gt_files = [p for p in sorted(gt_path.rglob("*")) if p.is_file() and p.suffix.lower() in valid_exts]

    if max_images is not None and max_images > 0:
        gt_files = gt_files[:max_images]

    if not gt_files:
        raise FileNotFoundError(f"No valid GT images found in {gt_path}")

    total_created = 0

    for idx, fpath in enumerate(gt_files):
        # Load image
        if fpath.suffix.lower() == ".npy":
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            # Handle normalized [-1, 1] range if present
            if arr.min() < -0.1:
                arr = arr * 0.5 + 0.5
        else:
            pil_img = Image.open(fpath).convert("L")
            arr = np.array(pil_img, dtype=np.float32) / 255.0

        arr = np.clip(arr, 0.0, 1.0)

        # Generate noise for each requested type
        for n_idx, n_type in enumerate(noise_types):
            item_seed = seed + idx * 100 + n_idx
            noisy_arr = generate_noise(arr, noise_type=n_type, intensity=intensity, seed=item_seed)

            sub_dir = out_path / n_type
            sub_dir.mkdir(parents=True, exist_ok=True)

            out_file = sub_dir / f"{fpath.stem}_{n_type}.png"
            uint8 = (np.clip(noisy_arr, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(uint8, mode="L").save(out_file)
            total_created += 1

    logger.info("Exported %d noisy benchmark images to %s", total_created, out_path)
    return total_created, out_path.resolve()
