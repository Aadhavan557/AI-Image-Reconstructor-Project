"""
phase10_inference.py
====================
AMSR-Net - Phase 10: Inference & Testing Pipeline
-------------------------------------------------
This script loads the trained AMSR-Net weights and runs the model
on completely unseen noisy images from the test dataset.

It performs:
1. Model instantiation and checkpoint loading.
2. Loading of unseen NoisyLR images.
3. Fast mixed-precision forward pass.
4. Denormalisation and side-by-side .png saving for visual inspection.
"""

import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import config as cfg
from dataset.dataloader import build_test_dataloader
from dataset.semiconductor_dataset import denormalise
from models.amsr_net import AMSRNet


def save_image_side_by_side(
    lr_tensor: torch.Tensor,
    sr_tensor: torch.Tensor,
    filename: str,
    out_dir: str
) -> None:
    """
    Saves a side-by-side comparison of the bicubic upsampled LR image
    and the AMSR-Net super-resolved output.

    Parameters
    ----------
    lr_tensor : torch.Tensor  Shape (1, H, W) in [0, 1]
    sr_tensor : torch.Tensor  Shape (1, 2H, 2W) in [0, 1]
    filename  : str           Original filename (e.g. '001.npy')
    out_dir   : str           Directory to save .png images
    """
    # 1. Bicubic upscale LR to match SR resolution for side-by-side
    # lr_tensor is (1, H, W). Add batch dim -> (1, 1, H, W)
    lr_upsampled = torch.nn.functional.interpolate(
        lr_tensor.unsqueeze(0),
        scale_factor=cfg.SCALE,
        mode="bicubic",
        align_corners=False
    ).squeeze(0)  # Back to (1, 2H, 2W)

    # 2. Convert to Numpy uint8 [0, 255]
    lr_arr = (lr_upsampled.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    sr_arr = (sr_tensor.squeeze(0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # 3. Concatenate horizontally
    side_by_side = np.concatenate([lr_arr, sr_arr], axis=1)

    # 4. Save using PIL
    img = Image.fromarray(side_by_side, mode='L')
    
    # Change extension from .npy to .png
    base_name = os.path.splitext(os.path.basename(filename))[0]
    save_path = os.path.join(out_dir, f"{base_name}_comparison.png")
    
    img.save(save_path)


def main():
    print("======================================================================")
    print("  AMSR-Net - Phase 10: Inference Pipeline")
    print("======================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 1. Create Output Directory
    out_dir = os.path.join(cfg.OUTPUTS_DIR, "test_restored")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Saving results to: {out_dir}")

    # 2. Build Test DataLoader
    test_loader = build_test_dataloader(batch_size=1)
    print(f"  Found {len(test_loader)} test images.")

    # 3. Build Model & Load Weights
    print(f"\n  Building AMSR-Net...")
    model = AMSRNet(
        in_channels      = cfg.CHANNELS,
        dim              = cfg.AMSRNET_DIM,
        encoder_blocks   = cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks = cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks      = cfg.AMSRNET_SWIN_BLOCKS,
        window_size      = cfg.AMSRNET_WINDOW_SIZE,
        num_heads        = cfg.AMSRNET_NUM_HEADS,
    )
    
    ckpt_path = cfg.BEST_MODEL_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best weights not found at {ckpt_path}! Did training finish?")
        
    print(f"  Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle DDP / state_dict formatting safely
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # 4. Inference Loop
    print("\n  Starting Inference...")
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Processing Images", ncols=80):
            lr_img   = batch["lr"].to(device, non_blocking=True)
            filename = batch["filename"][0]  # batch_size is 1

            # Forward pass (Mixed Precision for speed)
            with torch.amp.autocast("cuda", enabled=True):
                sr_img = model(lr_img)

            # Denormalise [-1, 1] -> [0, 1]
            lr_img_01 = denormalise(lr_img[0]).clamp(0.0, 1.0)
            sr_img_01 = denormalise(sr_img[0]).clamp(0.0, 1.0)

            # Save Side-by-Side comparison
            save_image_side_by_side(lr_img_01, sr_img_01, filename, out_dir)

    print("\n======================================================================")
    print(f"  Inference Complete! All restored images saved to:")
    print(f"  {out_dir}")
    print("======================================================================")


if __name__ == "__main__":
    main()
