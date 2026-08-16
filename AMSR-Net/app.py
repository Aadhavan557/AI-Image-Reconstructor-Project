"""
app.py
======
AMSR-Net Interactive Web Dashboard
----------------------------------
Features:
  1. 🔬 Single Image Restoration & Strict Independent Step 1 / Step 2 Execution Engine
  2. 🧪 Synthetic Noise Lab & Instant AMSR-Net Denoising Benchmark
  3. 📊 Visual Analytics: Zoomed Crops, Residual Maps, Pixel Histograms & Metric Gains

Run:
    streamlit run AMSR-Net/app.py
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
    page_title="AMSR-Net | AI Restoration & Noisy vs Restored Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# STYLING
# ===========================================================================
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hero Title ─────────────────────────────────────── */
    .hero-wrap {
        background: linear-gradient(135deg, rgba(0,212,255,0.08) 0%, rgba(123,47,247,0.12) 100%);
        border: 1px solid rgba(123,47,247,0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem 1.6rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse at top left, rgba(0,212,255,0.12), transparent 60%),
                    radial-gradient(ellipse at bottom right, rgba(123,47,247,0.15), transparent 60%);
        pointer-events: none;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.3rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.6;
    }
    .badge-row { display:flex; gap:0.5rem; margin-top:1rem; flex-wrap:wrap; }
    .badge {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #cbd5e1;
        font-weight: 600;
    }

    /* ── Step Cards ─────────────────────────────────────── */
    .step-card {
        background: rgba(30,30,46,0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem 1rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }
    .step-card:hover { border-color: rgba(0,212,255,0.35); }
    .step-num {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #00d4ff;
        margin-bottom: 0.25rem;
    }
    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 0.8rem;
    }
    .step-card-gt .step-num  { color: #a78bfa; }
    .step-card-gt { border-color: rgba(167,139,250,0.15); }
    .step-card-gt:hover { border-color: rgba(167,139,250,0.4); }

    /* ── Metric Cards ───────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,30,46,0.9), rgba(20,20,35,0.95));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.1rem 0.8rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); border-color: rgba(0,212,255,0.3); }
    .metric-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .metric-label { font-size: 0.72rem; color: #64748b; margin-top: 0.25rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    .gain-badge {
        display: inline-block;
        margin-top: 0.4rem;
        background: rgba(168,255,120,0.12);
        border: 1px solid rgba(168,255,120,0.3);
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        color: #a8ff78;
        font-weight: 700;
    }

    /* ── Section Headers ────────────────────────────────── */
    .section-header {
        display: flex; align-items: center; gap: 0.6rem;
        border-left: 3px solid #00d4ff;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.75rem;
    }
    .section-header h4 { margin: 0; font-size: 1rem; font-weight: 700; color: #e2e8f0; }
    .section-header .sh-icon { font-size: 1.2rem; }

    /* ── Info / Status Boxes ────────────────────────────── */
    .info-box {
        background: rgba(0,212,255,0.07);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .success-box {
        background: rgba(168,255,120,0.07);
        border: 1px solid rgba(168,255,120,0.25);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        color: #a8ff78;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* ── Buttons ────────────────────────────────────────── */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.65rem 1.4rem;
        width: 100%;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 20px rgba(0,212,255,0.25);
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 28px rgba(0,212,255,0.4);
        transform: translateY(-1px);
    }
    .stButton>button:active { transform: translateY(0); }

    /* ── Tabs ───────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,47,247,0.2)) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(0,212,255,0.3);
    }

    /* ── Divider ────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.07) !important; }

    /* ── Upload area ────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
        border: 1.5px dashed rgba(255,255,255,0.15) !important;
        background: rgba(255,255,255,0.02);
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(0,212,255,0.4) !important; }
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
    uploaded_file.seek(0)   # Reset pointer so the file can be read multiple times
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
    pred_arr = pred_t[0, 0].numpy()
    with torch.no_grad():
        psnr_val = psnr_fn(pred_t, gt_t).item()
        ssim_val = ssim_fn(pred_t, gt_t).item()

    mse_val = float(np.mean((pred_arr - gt_arr) ** 2))
    mae_val = float(np.mean(np.abs(pred_arr - gt_arr)))

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


def plot_histogram_comparison(noisy: np.ndarray, restored: np.ndarray, gt: Optional[np.ndarray] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#1e1e2e")

    n_hist, n_bins = np.histogram(noisy, bins=256, range=(0, 1))
    ax.plot(n_bins[:-1], n_hist, color="#ff6b6b", alpha=0.75, label="Noisy Input", linewidth=1.5)

    r_hist, r_bins = np.histogram(restored, bins=256, range=(0, 1))
    ax.plot(r_bins[:-1], r_hist, color="#00d4ff", alpha=0.9, label="Restored (AMSR-Net)", linewidth=2.0)

    if gt is not None:
        g_hist, g_bins = np.histogram(gt, bins=256, range=(0, 1))
        ax.plot(g_bins[:-1], g_hist, color="#a8ff78", alpha=0.65, label="Ground Truth", linestyle="--", linewidth=1.5)

    ax.set_title("Pixel Intensity Distribution Overlay", color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel Value (0.0 - 1.0)", color="#aaa", fontsize=9)
    ax.set_ylabel("Pixel Frequency", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.legend(facecolor="#0e1117", edgecolor="#333", labelcolor="white", fontsize=8)
    ax.grid(True, color="#333344", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def render_comparison_suite(noisy_arr: np.ndarray, restored_arr: np.ndarray, gt_arr: Optional[np.ndarray] = None):
    st.markdown(
        '<div class="section-header"><span class="sh-icon">🔍</span><h4>Deep Inspection Suite — Noisy vs Restored</h4></div>',
        unsafe_allow_html=True,
    )

    # Resample noisy_arr to match restored_arr spatial resolution if shapes differ (e.g. 2x Super-Resolution)
    if noisy_arr.shape != restored_arr.shape:
        pil_n = Image.fromarray((np.clip(noisy_arr, 0, 1) * 255).astype(np.uint8))
        pil_n_up = pil_n.resize((restored_arr.shape[1], restored_arr.shape[0]), Image.BICUBIC)
        noisy_proc = np.array(pil_n_up, dtype=np.float32) / 255.0
    else:
        noisy_proc = noisy_arr

    if gt_arr is not None and gt_arr.shape != restored_arr.shape:
        pil_g = Image.fromarray((np.clip(gt_arr, 0, 1) * 255).astype(np.uint8))
        pil_g_up = pil_g.resize((restored_arr.shape[1], restored_arr.shape[0]), Image.BICUBIC)
        gt_proc = np.array(pil_g_up, dtype=np.float32) / 255.0
    else:
        gt_proc = gt_arr

    # 1. High-Contrast Edge Crops
    st.markdown('<div class="section-header"><span class="sh-icon">1️⃣</span><h4>Zoomed Center-Crop Detail Patch</h4></div>', unsafe_allow_html=True)
    h, w = restored_arr.shape
    cy, cx = h // 2, w // 2
    ph, pw = min(64, h // 2), min(64, w // 2)

    n_crop = noisy_proc[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
    r_crop = restored_arr[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]

    ncols = 3 if gt_proc is not None else 2
    zcols = st.columns(ncols)
    with zcols[0]:
        st.image(n_crop, caption="Noisy Input Patch", use_container_width=True)
    with zcols[1]:
        st.image(r_crop, caption="AMSR-Net Restored Patch", use_container_width=True)
    if gt_proc is not None:
        gt_crop = gt_proc[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
        with zcols[2]:
            st.image(gt_crop, caption="Ground Truth Patch", use_container_width=True)

    # 2. Residual Maps & Heatmaps
    st.divider()
    st.markdown("#### 2️⃣ Removed Noise Residual Map (|Restored - Noisy|)")
    st.caption("Bright regions show where AMSR-Net performed high-magnitude noise suppression.")
    diff_rn = np.abs(restored_arr - noisy_proc)

    hcols = st.columns(2)
    with hcols[0]:
        st.image(diff_rn / (diff_rn.max() + 1e-8), caption=f"Absolute Noise Difference (Max Spike: {diff_rn.max():.4f})", use_container_width=True)
    with hcols[1]:
        fig_r, ax_r = plt.subplots(figsize=(5, 4), facecolor="#0e1117")
        ax_r.set_facecolor("#0e1117")
        im_r = ax_r.imshow(diff_rn, cmap="hot", vmin=0, vmax=max(float(diff_rn.max()), 1e-6))
        plt.colorbar(im_r, ax=ax_r, label="|Restored - Noisy|")
        ax_r.set_title("Removed Noise Heatmap", color="white")
        ax_r.axis("off")
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)

    # 3. Histogram Distribution Overlay
    st.divider()
    st.markdown("#### 3️⃣ Intensity Histogram Overlay")
    st.caption("Compare how pixel brightness distribution changes before vs after AMSR-Net restoration.")
    fig_hist = plot_histogram_comparison(noisy_arr, restored_arr, gt_arr)
    st.pyplot(fig_hist, use_container_width=True)


# ===========================================================================
# SIDEBAR
# ===========================================================================
def render_sidebar(gpu_name: str, device: torch.device):
    with st.sidebar:
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,rgba(0,212,255,0.1),rgba(123,47,247,0.1));
                        border:1px solid rgba(0,212,255,0.2);border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem;">
                <div style="font-size:1.3rem;font-weight:900;background:linear-gradient(135deg,#00d4ff,#a78bfa);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🔬 AMSR-Net</div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:0.2rem;font-weight:600;letter-spacing:0.5px;">AI SEMICONDUCTOR RESTORATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background:rgba(30,30,46,0.8);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1rem;">
                <div style="font-size:0.7rem;font-weight:700;color:#64748b;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.6rem;">⚙️ System Status</div>
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;">
                    <span style="width:8px;height:8px;border-radius:50%;background:#a8ff78;display:inline-block;"></span>
                    <span style="font-size:0.82rem;color:#e2e8f0;font-weight:600;">Device: {str(device).upper()}</span>
                </div>
                <div style="font-size:0.8rem;color:#94a3b8;padding-left:1.1rem;">{gpu_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.78rem;color:#475569;padding:0.5rem 0;line-height:1.8;">
                <b style="color:#94a3b8;">📌 How to use:</b><br>
                1. Upload your <b style="color:#00d4ff;">noisy SEM image</b> in Step 1<br>
                2. Click <b style="color:#00d4ff;">Restore</b> to run AI inference<br>
                3. (Optional) Upload a <b style="color:#a78bfa;">clean reference</b> in Step 2 for accuracy metrics
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# MAIN APP
# ===========================================================================
def main():
    model, device, gpu_name = load_model()
    render_sidebar(gpu_name, device)

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="main-title">🔬 AMSR-Net Dashboard</div>
            <div class="subtitle">AI-powered semiconductor image denoising &amp; 2× super-resolution —
            upload your SEM scan and get a clean, high-resolution restoration in seconds.</div>
            <div class="badge-row">
                <span class="badge">⚡ GPU Accelerated</span>
                <span class="badge">📊 PSNR · SSIM · MSE · MAE</span>
                <span class="badge">🧪 Synthetic Noise Lab</span>
                <span class="badge">⬇️ Export PNG / NPY</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(["🔬  Restore Image", "🧪  Synthetic Noise Lab"])

    # =======================================================================
    # TAB 1: SINGLE IMAGE RESTORATION
    # =======================================================================
    with tab1:
        if model is None:
            st.markdown('<div class="info-box">⚠️ Model weights not found at <code>weights/amsr_net_best.pth</code>. Inference is disabled.</div>', unsafe_allow_html=True)
        else:
            col_u1, col_u2 = st.columns(2, gap="large")

            # Step 1 Column
            with col_u1:
                st.markdown(
                    '<div class="step-card"><div class="step-num">Step 1 — Required</div>'
                    '<div class="step-title">📌 Upload Noisy / Corrupted Image</div></div>',
                    unsafe_allow_html=True,
                )
                deg_file = st.file_uploader(
                    "Upload Noisy Image",
                    type=["png", "jpg", "jpeg", "tiff", "npy", "tif"],
                    key="t1_deg",
                    label_visibility="collapsed",
                )
                btn_step1 = False
                if deg_file is not None:
                    btn_step1 = st.button("🚀 Restore with AMSR-Net", key="btn_step1")

            # Step 2 Column
            with col_u2:
                st.markdown(
                    '<div class="step-card step-card-gt"><div class="step-num">Step 2 — Optional</div>'
                    '<div class="step-title">🎯 Upload Clean Ground Truth</div></div>',
                    unsafe_allow_html=True,
                )
                gt_file = st.file_uploader(
                    "Upload Clean Ground Truth Image",
                    type=["png", "jpg", "jpeg", "tiff", "npy", "tif"],
                    key="t1_gt",
                    label_visibility="collapsed",
                )
                btn_step2 = False
                if gt_file is not None:
                    btn_step2 = st.button("📊 Evaluate Accuracy Metrics", key="btn_step2")

            st.divider()

            # Case A: Neither file uploaded
            if deg_file is None and gt_file is None:
                st.info("👈 Upload your **Noisy Image in Step 1** to restore it, and optionally upload your **Clean Ground Truth in Step 2** to calculate PSNR/SSIM accuracy metrics.")

            # Case B: Only Step 2 (Ground Truth) is uploaded
            elif deg_file is None and gt_file is not None:
                gt_arr, gt_info = load_uploaded_image(gt_file)
                st.session_state["t1_gt_arr"] = gt_arr

                st.info("🎯 **Clean Ground Truth Reference Registered!** Now upload your **Noisy Image in Step 1** on the left to restore it and calculate PSNR/SSIM accuracy metrics.")

                st.markdown("#### 🎯 Step 2: Clean Ground Truth Reference Preview")
                st.image(gt_arr, caption=f"Ground Truth Image ({gt_info['shape'][1]}x{gt_info['shape'][0]})", use_container_width=True)

            # Case C: Step 1 (Noisy Image) is uploaded (with or without Step 2)
            elif deg_file is not None:
                file_key_deg = f"deg_{deg_file.name}_{deg_file.size}"
                file_key_gt = f"gt_{gt_file.name}_{gt_file.size}" if gt_file else "none"
                combined_key = f"{file_key_deg}__{file_key_gt}"

                is_new_run = st.session_state.get("t1_key") != combined_key

                if btn_step1 or btn_step2 or is_new_run:
                    deg_arr, _ = load_uploaded_image(deg_file)
                    # Always load GT fresh from the uploaded file
                    gt_arr = load_uploaded_image(gt_file)[0] if gt_file is not None else None

                    with st.spinner("Running AMSR-Net AI Inference on Step 1 Noisy Image..."):
                        restored_arr, inf_ms = run_model_inference(model, deg_arr, device)

                    st.session_state["t1_key"] = combined_key
                    st.session_state["t1_deg_arr"] = deg_arr
                    st.session_state["t1_gt_arr"] = gt_arr
                    st.session_state["t1_restored_arr"] = restored_arr
                    st.session_state["t1_inf_ms"] = inf_ms

                # Retrieve saved state
                deg_arr = st.session_state["t1_deg_arr"]
                restored_arr = st.session_state["t1_restored_arr"]
                inf_ms = st.session_state["t1_inf_ms"]

                # Always use the CURRENT live gt_file (not stale session state)
                # so metrics appear whenever GT is present, even if re-uploaded
                gt_arr = load_uploaded_image(gt_file)[0] if gt_file is not None else None
                # Keep session state in sync
                st.session_state["t1_gt_arr"] = gt_arr

                st.success(f"✅ AMSR-Net Restoration Completed in {inf_ms:.1f} ms!")

                # Side-by-Side Visual Grid
                ncols = 3 if gt_arr is not None else 2
                cols = st.columns(ncols)
                with cols[0]:
                    st.markdown("#### 📌 Step 1: Noisy Input Image")
                    st.image(deg_arr, use_container_width=True)
                with cols[1]:
                    st.markdown("#### ✨ Restored Output (AMSR-Net)")
                    st.image(restored_arr, use_container_width=True)
                if gt_arr is not None:
                    with cols[2]:
                        st.markdown("#### 🎯 Step 2: Clean Ground Truth")
                        st.image(gt_arr, use_container_width=True)


                # ===========================================================
                # 📊 METRICS — Always shown after Step 1 restoration
                # ===========================================================
                st.divider()
                if gt_arr is not None:
                    st.markdown("#### 📊 Restoration Quality Metrics vs Ground Truth")
                    st.caption("Comparing AMSR-Net restored output against the uploaded clean ground truth.")
                    m_noisy = compute_metrics(deg_arr, gt_arr)
                    m_restored = compute_metrics(restored_arr, gt_arr)
                    psnr_gain = m_restored["PSNR (dB)"] - m_noisy["PSNR (dB)"]
                    ssim_gain = m_restored["SSIM"] - m_noisy["SSIM"]

                    mcols = st.columns(4)
                    with mcols[0]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m_restored["PSNR (dB)"]:.2f} dB</div>'
                            f'<div class="metric-label">Restored PSNR (vs GT)</div>'
                            f'<div class="gain-badge">+{psnr_gain:.2f} dB over noisy</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[1]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m_restored["SSIM"]:.4f}</div>'
                            f'<div class="metric-label">Restored SSIM (vs GT)</div>'
                            f'<div class="gain-badge">+{ssim_gain:.4f} over noisy</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[2]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m_restored["MSE"]:.6f}</div>'
                            f'<div class="metric-label">Restored MSE (vs GT)</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[3]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m_restored["MAE"]:.4f}</div>'
                            f'<div class="metric-label">Restored MAE (vs GT)</div></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    # No GT — show Noisy vs Restored metrics directly
                    st.markdown("#### 📊 Restoration Quality Metrics (Noisy vs Restored)")
                    st.caption("Comparing AMSR-Net restored output directly against the noisy input. Upload a **Clean Ground Truth in Step 2** for accuracy vs reference.")
                    m = compute_metrics(restored_arr, deg_arr)

                    mcols = st.columns(4)
                    with mcols[0]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m["PSNR (dB)"]:.2f} dB</div>'
                            f'<div class="metric-label">PSNR (Restored vs Noisy)</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[1]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m["SSIM"]:.4f}</div>'
                            f'<div class="metric-label">SSIM (Restored vs Noisy)</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[2]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m["MSE"]:.6f}</div>'
                            f'<div class="metric-label">MSE (Restored vs Noisy)</div></div>',
                            unsafe_allow_html=True,
                        )
                    with mcols[3]:
                        st.markdown(
                            f'<div class="metric-card"><div class="metric-value">{m["MAE"]:.4f}</div>'
                            f'<div class="metric-label">MAE (Restored vs Noisy)</div></div>',
                            unsafe_allow_html=True,
                        )


                # Deep Comparison Suite
                st.divider()
                render_comparison_suite(deg_arr, restored_arr, gt_arr)

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
        st.markdown("### 🧪 Synthetic Noise Lab & Instant AMSR-Net Denoising Benchmark")
        st.markdown(
            "Generate synthetic noise (Gaussian, Poisson, S&P, Speckle, Mixed SEM) and immediately "
            "run AMSR-Net restoration to evaluate denoising performance!"
        )

        n_col1, n_col2 = st.columns([1, 2])

        with n_col1:
            st.markdown("#### ⚙️ Noise Parameters")
            gt_upload = st.file_uploader("Upload Clean Ground Truth Image", type=["png", "jpg", "jpeg", "tiff", "npy", "tif"], key="t2_gt")

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
            st.markdown("#### 🖼️ Noisy vs Restored Live Comparison")
            if gt_upload is None:
                st.warning("👈 Upload a clean Ground Truth image to apply noise.")
            else:
                clean_arr, c_info = load_uploaded_image(gt_upload)
                noisy_arr = generate_noise(clean_arr, noise_type=noise_type, intensity=intensity, seed=seed)

                # Show GT vs Noisy preview
                vcols = st.columns(2)
                with vcols[0]:
                    st.markdown("##### 🎯 Clean Ground Truth")
                    st.image(clean_arr, use_container_width=True)
                with vcols[1]:
                    st.markdown(f"##### 📌 Corrupted Noisy ({desc['name']})")
                    st.image(noisy_arr, use_container_width=True)

                st.divider()
                st.markdown("#### 🚀 AMSR-Net Live Restoration Test")

                if model is None:
                    st.warning("⚠️ Model weights not found. Cannot run live restoration.")
                else:
                    t2_key = f"t2_{gt_upload.name}_{gt_upload.size}_{noise_type}_{intensity}_{seed}"
                    restore_t2_clicked = st.button("✨ RUN AMSR-NET RESTORATION & COMPARE", key="t2_restore_btn")

                    if restore_t2_clicked or st.session_state.get("t2_key") == t2_key:
                        if restore_t2_clicked or "t2_synth_restored" not in st.session_state or st.session_state.get("t2_key") != t2_key:
                            with st.spinner("Restoring synthetic noise..."):
                                synth_restored, s_ms = run_model_inference(model, noisy_arr, device)

                            st.session_state["t2_key"] = t2_key
                            st.session_state["t2_synth_restored"] = synth_restored
                            st.session_state["t2_s_ms"] = s_ms

                        synth_restored = st.session_state["t2_synth_restored"]
                        s_ms = st.session_state["t2_s_ms"]

                        st.success(f"✅ Restored synthetic {desc['name']} noise in {s_ms:.1f} ms!")

                        # 3-Column Visual: GT | Noisy | Restored
                        rcols = st.columns(3)
                        with rcols[0]:
                            st.markdown("##### 🎯 Clean GT")
                            st.image(clean_arr, use_container_width=True)
                        with rcols[1]:
                            st.markdown("##### 📌 Synthesized Noisy")
                            st.image(noisy_arr, use_container_width=True)
                        with rcols[2]:
                            st.markdown("##### ✨ AMSR-Net Restored")
                            st.image(synth_restored, use_container_width=True)

                        # Denoising Metrics & Gain
                        m_n = compute_metrics(noisy_arr, clean_arr)
                        m_r = compute_metrics(synth_restored, clean_arr)
                        p_gain = m_r["PSNR (dB)"] - m_n["PSNR (dB)"]
                        s_gain = m_r["SSIM"] - m_n["SSIM"]

                        st.divider()
                        st.markdown("#### 📊 Denoising Gain Breakdown")
                        gcols = st.columns(4)
                        with gcols[0]:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{m_r["PSNR (dB)"]:.2f} dB</div>'
                                f'<div class="metric-label">Restored PSNR</div>'
                                f'<div class="gain-badge">+{p_gain:.2f} dB Gain</div></div>',
                                unsafe_allow_html=True,
                            )
                        with gcols[1]:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{m_r["SSIM"]:.4f}</div>'
                                f'<div class="metric-label">Restored SSIM</div>'
                                f'<div class="gain-badge">+{s_gain:.4f} Gain</div></div>',
                                unsafe_allow_html=True,
                            )
                        with gcols[2]:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{m_n["PSNR (dB)"]:.2f} dB</div>'
                                f'<div class="metric-label">Initial Noisy PSNR</div></div>',
                                unsafe_allow_html=True,
                            )
                        with gcols[3]:
                            st.markdown(
                                f'<div class="metric-card"><div class="metric-value">{m_n["SSIM"]:.4f}</div>'
                                f'<div class="metric-label">Initial Noisy SSIM</div></div>',
                                unsafe_allow_html=True,
                            )

                        # Render deep comparison suite
                        st.divider()
                        render_comparison_suite(noisy_arr, synth_restored, clean_arr)

                st.divider()
                st.markdown("#### 📦 Export Benchmark Test Package")
                st.caption("Click below to create a ZIP bundle containing all 5 noise variations of your uploaded image!")

                if st.button("🎁 Export Full 5-Noise ZIP Bundle", key="zip_btn"):
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w") as zf:
                        zf.writestr("clean_gt.png", arr_to_png_bytes(clean_arr))
                        for n_k in ["gaussian", "poisson", "salt_pepper", "speckle", "mixed_sem"]:
                            n_arr = generate_noise(clean_arr, noise_type=n_k, intensity=intensity, seed=seed)
                            zf.writestr(f"test_noisy_{n_k}.png", arr_to_png_bytes(n_arr))

                    st.download_button(
                        label="⬇️ Download test_benchmark_suite.zip",
                        data=zip_buf.getvalue(),
                        file_name="test_benchmark_suite.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
