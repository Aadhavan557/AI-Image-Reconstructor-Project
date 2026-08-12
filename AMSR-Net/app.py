"""
app.py
======
AMSR-Net Interactive Web Dashboard
----------------------------------
Features:
  1. 🔬 Single Image Restoration (AMSR-Net GPU Inference)
  2. 🧪 Synthetic Noise Lab (Gaussian, Poisson, S&P, Speckle, Mixed SEM) & Benchmark Suite Exporter
  3. ⚔️ Dual-Model Arena (Compare Your Model vs Friend's Model Outputs & Identify Winner)

Run:
    streamlit run app.py
"""

import io
import logging
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg
from dataset.semiconductor_dataset import denormalise, normalise_lr
from metrics.psnr import PSNRMetric
from metrics.ssim_metric import SSIMMetric
from models.amsr_net import AMSRNet
from utils.noise_generator import (
    NOISE_DESCRIPTIONS,
    generate_noise,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="AMSR-Net | AI Restoration & Model Arena",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# STYLING
# ===========================================================================
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #888;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
    }
    .winner-box-m1 {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(123, 47, 247, 0.2));
        border: 2px solid #00d4ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .winner-box-m2 {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15), rgba(255, 179, 71, 0.2));
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .winner-box-tie {
        background: linear-gradient(135deg, rgba(168, 255, 120, 0.15), rgba(120, 255, 214, 0.2));
        border: 2px solid #a8ff78;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .w-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    .w-sub {
        font-size: 1.0rem;
        color: #dddddd;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.5rem 1.5rem;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.88;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ===========================================================================
# MODEL LOADING (Cached)
# ===========================================================================
@st.cache_resource(show_spinner="Loading AMSR-Net weights...")
def load_model() -> Tuple[Optional[AMSRNet], torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    ckpt_path = Path(cfg.BEST_MODEL_PATH)
    if not ckpt_path.exists():
        return None, device, gpu_name

    model = AMSRNet(
        in_channels=cfg.CHANNELS,
        dim=cfg.AMSRNET_DIM,
        encoder_blocks=cfg.AMSRNET_ENCODER_BLOCKS,
        restormer_blocks=cfg.AMSRNET_RESTORMER_BLOCKS,
        swin_blocks=cfg.AMSRNET_SWIN_BLOCKS,
        window_size=cfg.AMSRNET_WINDOW_SIZE,
        num_heads=cfg.AMSRNET_NUM_HEADS,
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device, gpu_name


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================
def load_uploaded_image(uploaded_file) -> Tuple[np.ndarray, dict]:
    raw_bytes = uploaded_file.read()
    ext = Path(uploaded_file.name).suffix.lower()

    if ext == ".npy":
        arr = np.load(io.BytesIO(raw_bytes)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
    else:
        pil = Image.open(io.BytesIO(raw_bytes)).convert("L")
        arr = np.array(pil, dtype=np.float32) / 255.0

    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Invalid image dimensions. Expected 2D grayscale image.")

    arr = np.clip(arr, 0.0, 1.0)
    info = {"shape": arr.shape, "min": float(arr.min()), "max": float(arr.max())}
    return arr, info


def run_model_inference(model: AMSRNet, arr: np.ndarray, device: torch.device) -> Tuple[np.ndarray, float]:
    norm_arr = normalise_lr(arr)
    tensor = torch.from_numpy(norm_arr[np.newaxis, np.newaxis, ...]).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    out_01 = denormalise(out[0, 0]).clamp(0.0, 1.0).float().cpu().numpy()
    return out_01, (t1 - t0) * 1000.0


@st.cache_resource
def get_metrics_calculators():
    return PSNRMetric(data_range=1.0), SSIMMetric(data_range=1.0, channels=1)


def compute_metrics(restored: np.ndarray, gt: np.ndarray) -> dict:
    psnr_fn, ssim_fn = get_metrics_calculators()
    pred_t = torch.from_numpy(restored).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)

    if pred_t.shape != gt_t.shape:
        gt_t = F.interpolate(gt_t, size=pred_t.shape[-2:], mode="bilinear", align_corners=False)

    gt_arr = gt_t[0, 0].numpy()
    with torch.no_grad():
        psnr_val = psnr_fn(pred_t, gt_t).item()
        ssim_val = ssim_fn(pred_t, gt_t).item()

    mse_val = float(np.mean((restored - gt_arr) ** 2))
    mae_val = float(np.mean(np.abs(restored - gt_arr)))

    # Sobel Edge Error
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ep = torch.sqrt(F.conv2d(pred_t, kx, padding=1) ** 2 + F.conv2d(pred_t, ky, padding=1) ** 2)
    eg = torch.sqrt(F.conv2d(gt_t, kx, padding=1) ** 2 + F.conv2d(gt_t, ky, padding=1) ** 2)
    edge_err = float(torch.mean(torch.abs(ep - eg)).item())

    return {"PSNR (dB)": psnr_val, "SSIM": ssim_val, "MSE": mse_val, "MAE": mae_val, "Edge Error": edge_err}


def arr_to_png_bytes(arr: np.ndarray) -> bytes:
    uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(uint8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def sobel_edge(arr: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    edge = torch.sqrt(gx**2 + gy**2)[0, 0].numpy()
    return (edge / (edge.max() + 1e-8)).astype(np.float32)


# ===========================================================================
# SIDEBAR
# ===========================================================================
def render_sidebar(gpu_name: str, device: torch.device):
    with st.sidebar:
        st.markdown("### 🔬 AMSR-Net Dashboard")
        st.markdown("**AI Semiconductor Image Restoration**")
        st.divider()
        st.markdown("#### ⚙️ System Status")
        st.info(f"**Device:** {str(device).upper()}\n\n**GPU:** {gpu_name}")
        st.markdown("#### ⚔️ Dual-Model Arena")
        st.caption("Compare your model vs your friend's model on test noise images!")
        st.divider()
        st.caption("Phase 11 — Comprehensive Benchmark Engine")


# ===========================================================================
# MAIN APP
# ===========================================================================
def main():
    model, device, gpu_name = load_model()
    render_sidebar(gpu_name, device)

    st.markdown('<p class="main-title">🔬 AMSR-Net & Model Comparison Arena</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">AI Semiconductor Image Denoising + 2× Super-Resolution · '
        'Synthetic Noise Benchmark Suite · Dual-Model Comparison Engine</p>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🔬 1. Single Image Restoration", "🧪 2. Synthetic Noise Lab", "⚔️ 3. Dual-Model Arena"])

    # =======================================================================
    # TAB 1: SINGLE IMAGE RESTORATION
    # =======================================================================
    with tab1:
        st.markdown("### 🖼️ Single Image Restoration & Inference")
        if model is None:
            st.warning("⚠️ Model weights (`weights/amsr_net_best.pth`) not found. Inference mode disabled.")
        else:
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                deg_file = st.file_uploader("Upload Degraded Image (NoisyLR)", type=["png", "jpg", "jpeg", "tiff", "npy"], key="t1_deg")
            with col_u2:
                gt_file = st.file_uploader("Upload Ground Truth (Optional)", type=["png", "jpg", "jpeg", "tiff", "npy"], key="t1_gt")

            if deg_file is not None:
                deg_arr, deg_info = load_uploaded_image(deg_file)
                gt_arr = load_uploaded_image(gt_file)[0] if gt_file is not None else None

                if st.button("🚀 RESTORE WITH AMSR-NET", key="t1_btn"):
                    with st.spinner("Running AMSR-Net..."):
                        restored_arr, inf_ms = run_model_inference(model, deg_arr, device)

                    st.success(f"✅ Restored in {inf_ms:.1f} ms!")

                    # Side by side visual
                    ncols = 3 if gt_arr is not None else 2
                    cols = st.columns(ncols)
                    with cols[0]:
                        st.markdown("#### 📌 Noisy Input")
                        st.image(deg_arr, use_container_width=True)
                    with cols[1]:
                        st.markdown("#### ✨ Restored (AMSR-Net)")
                        st.image(restored_arr, use_container_width=True)
                    if gt_arr is not None:
                        with cols[2]:
                            st.markdown("#### 🎯 Ground Truth")
                            st.image(gt_arr, use_container_width=True)

                    # Metrics if GT
                    if gt_arr is not None:
                        st.divider()
                        st.markdown("#### 📊 Evaluation Metrics")
                        m = compute_metrics(restored_arr, gt_arr)
                        mcols = st.columns(4)
                        for col, (k, v) in zip(mcols, m.items()):
                            if k == "Edge Error":
                                continue
                            with col:
                                fmt_v = f"{v:.4f}" if k in ("SSIM", "MSE", "MAE") else f"{v:.2f}"
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-value">{fmt_v}</div>'
                                    f'<div class="metric-label">{k}</div></div>',
                                    unsafe_allow_html=True,
                                )

                    # Downloads
                    st.divider()
                    dcol1, dcol2 = st.columns(2)
                    with dcol1:
                        st.download_button("⬇️ Download Restored PNG", arr_to_png_bytes(restored_arr), "restored.png", "image/png", use_container_width=True)
                    with dcol2:
                        npy_buf = io.BytesIO()
                        np.save(npy_buf, restored_arr)
                        st.download_button("⬇️ Download Restored NPY", npy_buf.getvalue(), "restored.npy", "application/octet-stream", use_container_width=True)

    # =======================================================================
    # TAB 2: SYNTHETIC NOISE LAB
    # =======================================================================
    with tab2:
        st.markdown("### 🧪 Synthetic Noise Lab & Benchmark Exporter")
        st.markdown(
            "Generate standard computer vision & electron microscopy noise (Gaussian, Poisson, S&P, Speckle, Mixed) "
            "to stress-test models or create a test set for your friend!"
        )

        n_col1, n_col2 = st.columns([1, 2])

        with n_col1:
            st.markdown("#### ⚙️ Noise Parameters")
            gt_upload = st.file_uploader("Upload Clean Ground Truth Image", type=["png", "jpg", "jpeg", "tiff", "npy"], key="t2_gt")

            noise_type = st.selectbox(
                "Select Noise Type",
                options=["gaussian", "poisson", "salt_pepper", "speckle", "mixed_sem"],
                format_func=lambda x: NOISE_DESCRIPTIONS[x]["name"],
            )

            intensity = st.slider("Noise Intensity / Severity", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
            seed = st.number_input("Random Seed (for exact reproducibility)", min_value=0, max_value=9999, value=42)

            desc = NOISE_DESCRIPTIONS[noise_type]
            st.info(f"**{desc['name']}**\n\n{desc['description']}\n\n*{desc['params']}*")

        with n_col2:
            st.markdown("#### 🖼️ Noisy Output Preview")
            if gt_upload is None:
                st.warning("👈 Upload a clean Ground Truth image to apply noise.")
            else:
                clean_arr, c_info = load_uploaded_image(gt_upload)
                noisy_arr = generate_noise(clean_arr, noise_type=noise_type, intensity=intensity, seed=seed)

                vcols = st.columns(2)
                with vcols[0]:
                    st.markdown("##### Clean Ground Truth")
                    st.image(clean_arr, use_container_width=True)
                with vcols[1]:
                    st.markdown(f"##### Corrupted ({desc['name']})")
                    st.image(noisy_arr, use_container_width=True)

                # Difference Residual
                diff_map = np.abs(noisy_arr - clean_arr)
                st.markdown("##### 🔍 Added Noise Residual (|Noisy - Clean|)")
                st.image(diff_map / (diff_map.max() + 1e-8), caption=f"Max Noise Spike: {diff_map.max():.4f}", use_container_width=True)

                st.divider()
                st.markdown("#### 📦 Export Test Package for Your Friend")
                st.caption("Click below to create a ZIP bundle containing all 5 noise variations of your uploaded image!")

                if st.button("🎁 Export Full 5-Noise ZIP Bundle", key="zip_btn"):
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        # Add GT
                        zf.writestr("clean_gt.png", arr_to_png_bytes(clean_arr))

                        # Add all noise types
                        for n_k in ["gaussian", "poisson", "salt_pepper", "speckle", "mixed_sem"]:
                            n_arr = generate_noise(clean_arr, noise_type=n_k, intensity=intensity, seed=seed)
                            zf.writestr(f"test_noisy_{n_k}.png", arr_to_png_bytes(n_arr))

                    st.download_button(
                        label="⬇️ Download test_suite_for_friend.zip",
                        data=zip_buf.getvalue(),
                        file_name="test_suite_for_friend.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

    # =======================================================================
    # TAB 3: DUAL-MODEL ARENA (Option 2 — Image Output Comparison)
    # =======================================================================
    with tab3:
        st.markdown("### ⚔️ Dual-Model Arena: Your Model vs Friend's Model")
        st.markdown(
            "Upload the Ground Truth clean image along with **Your Model's Restored Output** and "
            "**Your Friend's Model Restored Output** to compare quality metrics and identify the winning model!"
        )
        st.divider()

        # 3 Uploaders
        ucol1, ucol2, ucol3 = st.columns(3)
        with ucol1:
            gt_arena_file = st.file_uploader("1️⃣ Ground Truth Clean Image", type=["png", "jpg", "jpeg", "tiff", "npy"], key="a_gt")
        with ucol2:
            m1_arena_file = st.file_uploader("2️⃣ Your Model Restored Image (M1)", type=["png", "jpg", "jpeg", "tiff", "npy"], key="a_m1")
        with ucol3:
            m2_arena_file = st.file_uploader("3️⃣ Friend's Model Restored Image (M2)", type=["png", "jpg", "jpeg", "tiff", "npy"], key="a_m2")

        if gt_arena_file is None or m1_arena_file is None or m2_arena_file is None:
            st.info("👆 Upload all 3 images (Ground Truth, Your Model Output, and Friend's Model Output) to run the comparison arena.")
            return

        # Load images
        try:
            gt_a_arr = load_uploaded_image(gt_arena_file)[0]
            m1_a_arr = load_uploaded_image(m1_arena_file)[0]
            m2_a_arr = load_uploaded_image(m2_arena_file)[0]
        except Exception as e:
            st.error(f"❌ Error loading images: {e}")
            return

        # Compute Metrics
        m1_metrics = compute_metrics(m1_a_arr, gt_a_arr)
        m2_metrics = compute_metrics(m2_a_arr, gt_a_arr)

        psnr_diff = m1_metrics["PSNR (dB)"] - m2_metrics["PSNR (dB)"]
        ssim_diff = m1_metrics["SSIM"] - m2_metrics["SSIM"]

        # Declare Winner
        st.divider()
        if psnr_diff > 0.1 and ssim_diff > 0.002:
            st.markdown(
                f"""
                <div class="winner-box-m1">
                    <div class="w-title">🏆 WINNER: YOUR MODEL (Model 1)</div>
                    <div class="w-sub">Your model is superior! It achieved <b>+{psnr_diff:.2f} dB</b> higher PSNR and <b>+{ssim_diff:.4f}</b> higher SSIM.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif psnr_diff < -0.1 and ssim_diff < -0.002:
            st.markdown(
                f"""
                <div class="winner-box-m2">
                    <div class="w-title">🏆 WINNER: FRIEND'S MODEL (Model 2)</div>
                    <div class="w-sub">Friend's model wins on this test! It achieved <b>+{-psnr_diff:.2f} dB</b> higher PSNR and <b>+{-ssim_diff:.4f}</b> higher SSIM.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="winner-box-tie">
                    <div class="w-title">🤝 TIE / EQUAL PERFORMANCE</div>
                    <div class="w-sub">Both models perform almost identically (PSNR diff: {psnr_diff:+.2f} dB).</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Comparative Metrics Table
        st.markdown("#### 📊 Metric Benchmark Comparison")
        bcols = st.columns(4)
        for col, metric in zip(bcols, ["PSNR (dB)", "SSIM", "MSE", "MAE"]):
            v1 = m1_metrics[metric]
            v2 = m2_metrics[metric]

            if metric in ("PSNR (dB)", "SSIM"):
                best = "M1" if v1 > v2 else ("M2" if v2 > v1 else "Tie")
            else:
                best = "M1" if v1 < v2 else ("M2" if v2 < v1 else "Tie")

            fmt1 = f"{v1:.2f} dB" if metric == "PSNR (dB)" else f"{v1:.4f}"
            fmt2 = f"{v2:.2f} dB" if metric == "PSNR (dB)" else f"{v2:.4f}"

            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{metric}</div>
                        <div style="font-size:1.1rem; color:#00d4ff; font-weight:700;">Your Model: {fmt1}</div>
                        <div style="font-size:1.1rem; color:#ff6b6b; font-weight:700;">Friend: {fmt2}</div>
                        <div style="font-size:0.85rem; color:#a8ff78; margin-top:0.3rem;">Best: <b>{best}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Side by Side Visual Comparison
        st.divider()
        st.markdown("#### 📸 Side-by-Side Visual Comparison")
        acols = st.columns(3)
        with acols[0]:
            st.markdown("##### 🎯 Ground Truth")
            st.image(gt_a_arr, use_container_width=True)
        with acols[1]:
            st.markdown("##### ⚡ Your Model Output (M1)")
            st.image(m1_a_arr, use_container_width=True)
        with acols[2]:
            st.markdown("##### ⚡ Friend's Model Output (M2)")
            st.image(m2_a_arr, use_container_width=True)

        # Zoomed Crop Comparison
        st.markdown("#### 🔍 Zoomed High-Contrast Edge Crops")
        h, w = gt_a_arr.shape
        cy, cx = h // 2, w // 2
        ph, pw = min(64, h // 2), min(64, w // 2)

        gt_crop = gt_a_arr[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
        m1_crop = m1_a_arr[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
        m2_crop = m2_a_arr[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]

        zcols = st.columns(3)
        with zcols[0]:
            st.image(gt_crop, caption="GT Center Crop", use_container_width=True)
        with zcols[1]:
            st.image(m1_crop, caption="Your Model Crop", use_container_width=True)
        with zcols[2]:
            st.image(m2_crop, caption="Friend Model Crop", use_container_width=True)

        # Absolute Error Heatmaps
        st.divider()
        st.markdown("#### 🌡️ Absolute Error Heatmaps (|Model - GT|)")
        st.caption("Cooler colors (black/dark red) mean low error (higher accuracy). Bright yellow/white means high error.")

        err1 = np.abs(gt_a_arr - m1_a_arr)
        err2 = np.abs(gt_a_arr - m2_a_arr)
        max_err = max(float(err1.max()), float(err2.max()), 1e-6)

        ecols = st.columns(2)
        with ecols[0]:
            fig1, ax1 = plt.subplots(figsize=(5, 4.5), facecolor="#0e1117")
            ax1.set_facecolor("#0e1117")
            im1 = ax1.imshow(err1, cmap="hot", vmin=0, vmax=max_err)
            plt.colorbar(im1, ax=ax1, label="|GT - Your Model|")
            ax1.set_title("Your Model Error Map", color="white")
            ax1.axis("off")
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            st.caption(f"Mean Error: {err1.mean():.4f}")

        with ecols[1]:
            fig2, ax2 = plt.subplots(figsize=(5, 4.5), facecolor="#0e1117")
            ax2.set_facecolor("#0e1117")
            im2 = ax2.imshow(err2, cmap="hot", vmin=0, vmax=max_err)
            plt.colorbar(im2, ax=ax2, label="|GT - Friend Model|")
            ax2.set_title("Friend's Model Error Map", color="white")
            ax2.axis("off")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            st.caption(f"Mean Error: {err2.mean():.4f}")


if __name__ == "__main__":
    main()
