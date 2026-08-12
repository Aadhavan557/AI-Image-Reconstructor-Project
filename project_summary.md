# AI Image Reconstructor (AMSR-Net) Project Summary

This document outlines the detailed methodology used in your semiconductor image reconstruction project, a summary of what you have accomplished so far, and the exact steps we will take once your current training run finishes.

---

## 1. Which Method is Used in This Project?
The primary goal of this project is to take low-resolution, noisy electron microscope images of semiconductors and perform **Joint Denoising and 2x Super-Resolution (SR)**. 

To achieve this, we built a custom, state-of-the-art hybrid AI architecture from scratch called **AMSR-Net** (Adaptive Multi-Expert Semiconductor Restoration Network).

### The AMSR-Net Architecture
AMSR-Net is a "multi-expert" model that combines the three best computer vision paradigms into one lightweight model optimized for your RTX 2050 GPU:
1. **CNN Residual Blocks (The Local Expert)**: Fast convolutional layers that extract basic local features (lines, corners) and perform initial noise removal.
2. **Restormer Blocks (The Global Denoising Expert)**: Uses a special Transformer technique called *Multi-Dconv Head Transposed Attention (MDTA)*. Instead of looking at spatial pixels, it looks across the *channels* of the image. This allows it to capture global speckle noise patterns without running out of GPU memory.
3. **Swin Transformer Blocks (The Sharpening Expert)**: Uses *Window-based Multi-Head Self-Attention (W-MSA)*. It breaks the image into small windows and analyzes how the pixels relate to each other locally. This is incredibly powerful for sharpening the microscopic edges of semiconductor circuits.
4. **Global Residual Connection**: We use Bilinear Interpolation to stretch the original blurry image to 2x size, and simply add it to the network's output. Because of our **Zero-Initialization** technique, the network starts perfectly stable and only has to learn the high-frequency "corrections" (removing noise and sharpening edges).

### The Loss Functions
To train the model, we use a custom **Composite Loss** made of three parts:
- **Charbonnier Loss (50%)**: A smoothed mathematical curve that forces the model to match the exact pixel colors of the Ground Truth, but is robust to extreme noise outliers.
- **SSIM Loss (20%)**: Structural Similarity Index. It forces the AI to respect the physical "structure" of the image (circuit lines and gratings) rather than just looking at raw pixel colors.
- **Edge Loss (10%)**: Uses a Sobel filter to detect the edges in the image and heavily penalizes the AI if its reconstructed edges are blurry.

---

## 2. What Have You Done So Far?
You have successfully completed the hardest parts of the machine learning pipeline:
* **Data Pipeline (Phases 1-2)**: You organized the `NoisyLR` (blurry inputs) and `GT` (perfect targets) datasets, and built a custom PyTorch DataLoader to normalize the images into the `[-1, 1]` mathematical space.
* **Architecture Construction (Phases 3-7)**: You built the CNN, Restormer, and Swin Transformer blocks entirely from scratch and assembled them into AMSR-Net.
* **Bug Squashing (Phase 8)**: You navigated and fixed incredibly complex, advanced PyTorch bugs, including:
  * Fixing Windows OS memory crashes (`NUM_WORKERS=0`).
  * Fixing 16-bit Mixed Precision overflows that caused `NaN` (infinity) crashes in the loss functions.
  * Fixing the "Dead Residual Plateau" by meticulously zero-initializing the final network layer.
* **Model Training (Phase 9)**: You are currently running the final, fully-stabilized training run (`amsrnet_run5`). Your RTX 2050 GPU is actively learning how to reconstruct semiconductor images!

---

## 3. What Do We Do After Training?
Once your terminal finishes `amsrnet_run5` (or triggers Early Stopping), the absolute best weights the AI discovered will be permanently saved to `weights/amsr_net_best.pth`.

From there, we move to the final stages:

### Phase 10: Inference Script
We will write a new script (`phase10_inference.py`). This script will:
1. Load the `amsr_net_best.pth` AI brain.
2. Open a folder of completely unseen, real-world noisy images (e.g., `dataset/test/`).
3. Feed them into the AI to let it dynamically clean and upscale them to 2x resolution.
4. Save the beautiful, high-resolution outputs to a `results/` folder on your hard drive.

### Phase 11: Visual Evaluation & Deployment
You will physically open the images on your computer and compare the blurry inputs to the AI's outputs. You should see crisp, clean semiconductor edges with the speckle noise entirely removed.
If you are satisfied with the results, the AMSR-Net model is officially complete and ready to be integrated into an automated factory inspection backend!
