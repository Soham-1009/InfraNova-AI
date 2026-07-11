from __future__ import annotations

import hashlib
import io
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.utils import enhance_output, visualize_tir_as_thermal
from src.inference.landsat_inference import LandsatColorizationInference

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best" / "pix2pix_landsat_best.pth"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
PREVIEW_SIZE = 512
RESAMPLING = getattr(Image, "Resampling", Image)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="InfraNova AI — Thermal to RGB",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Theme system
# ---------------------------------------------------------------------------
def get_theme(light_mode: bool) -> Dict[str, str]:
    """Return a colour palette for the requested appearance."""
    if light_mode:
        return {
            "bg": "#f0f2f5",
            "bg_gradient": "linear-gradient(135deg, #f0f2f5 0%, #e8ecf1 100%)",
            "surface": "rgba(255, 255, 255, 0.72)",
            "surface_solid": "#ffffff",
            "surface_hover": "rgba(255, 255, 255, 0.88)",
            "text": "#0f1419",
            "text_secondary": "#536471",
            "border": "rgba(0, 0, 0, 0.08)",
            "border_hover": "rgba(0, 0, 0, 0.16)",
            "primary": "#0d9488",
            "primary_hover": "#0f766e",
            "primary_text": "#ffffff",
            "accent": "#d97706",
            "accent_soft": "rgba(217, 119, 6, 0.10)",
            "success": "#059669",
            "success_soft": "rgba(5, 150, 105, 0.12)",
            "danger": "#dc2626",
            "image_well": "#e5e7eb",
            "card_shadow": "0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)",
            "card_shadow_hover": "0 10px 25px rgba(0,0,0,0.08), 0 4px 10px rgba(0,0,0,0.04)",
            "glow": "0 0 20px rgba(13,148,136,0.15)",
            "gradient_primary": "linear-gradient(135deg, #0d9488 0%, #0284c7 100%)",
            "gradient_accent": "linear-gradient(135deg, #d97706 0%, #ea580c 100%)",
        }
    return {
        "bg": "#0a0e12",
        "bg_gradient": "linear-gradient(135deg, #0a0e12 0%, #0f1720 50%, #0a1628 100%)",
        "surface": "rgba(17, 25, 35, 0.65)",
        "surface_solid": "#111923",
        "surface_hover": "rgba(17, 25, 35, 0.85)",
        "text": "#e7edf3",
        "text_secondary": "#8899a6",
        "border": "rgba(255, 255, 255, 0.06)",
        "border_hover": "rgba(255, 255, 255, 0.14)",
        "primary": "#14b8a6",
        "primary_hover": "#2dd4bf",
        "primary_text": "#042f2e",
        "accent": "#f59e0b",
        "accent_soft": "rgba(245, 158, 11, 0.12)",
        "success": "#10b981",
        "success_soft": "rgba(16, 185, 129, 0.12)",
        "danger": "#ef4444",
        "image_well": "#0d1117",
        "card_shadow": "0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)",
        "card_shadow_hover": "0 10px 30px rgba(0,0,0,0.4), 0 4px 12px rgba(0,0,0,0.3)",
        "glow": "0 0 30px rgba(20,184,166,0.18)",
        "gradient_primary": "linear-gradient(135deg, #14b8a6 0%, #0ea5e9 100%)",
        "gradient_accent": "linear-gradient(135deg, #f59e0b 0%, #f97316 100%)",
    }


def inject_css(theme: Dict[str, str]) -> None:
    """Inject the full design system as a single <style> block."""
    css = f"""
    <style>
        /* ── Google Fonts ─────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* ── CSS Custom Properties ────────────────────────────── */
        :root {{
            --bg: {theme['bg']};
            --bg-gradient: {theme['bg_gradient']};
            --surface: {theme['surface']};
            --surface-solid: {theme['surface_solid']};
            --surface-hover: {theme['surface_hover']};
            --text: {theme['text']};
            --text-secondary: {theme['text_secondary']};
            --border: {theme['border']};
            --border-hover: {theme['border_hover']};
            --primary: {theme['primary']};
            --primary-hover: {theme['primary_hover']};
            --primary-text: {theme['primary_text']};
            --accent: {theme['accent']};
            --accent-soft: {theme['accent_soft']};
            --success: {theme['success']};
            --success-soft: {theme['success_soft']};
            --danger: {theme['danger']};
            --image-well: {theme['image_well']};
            --card-shadow: {theme['card_shadow']};
            --card-shadow-hover: {theme['card_shadow_hover']};
            --glow: {theme['glow']};
            --gradient-primary: {theme['gradient_primary']};
            --gradient-accent: {theme['gradient_accent']};
        }}

        /* ── Reset & Base ─────────────────────────────────────── */
        *, *::before, *::after {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }}

        #MainMenu, footer,
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        [data-testid="stAppViewContainer"], .stApp {{
            background: var(--bg-gradient) !important;
            color: var(--text) !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent !important;
        }}

        .block-container {{
            max-width: 1360px;
            padding: 1rem 1.5rem 2.5rem;
        }}

        /* Force all text to theme colour */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp li, .stApp label, .stApp span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] * {{
            color: var(--text) !important;
        }}

        /* ── Animations ───────────────────────────────────────── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes pulse-dot {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50%      {{ opacity: 0.5; transform: scale(1.3); }}
        }}
        @keyframes shimmer {{
            0%   {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        @keyframes gradientShift {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        /* ── Brand Header ─────────────────────────────────────── */
        .brand-row {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            animation: fadeInUp 0.5s ease-out;
        }}

        .brand-logo {{
            display: grid;
            place-items: center;
            width: 2.8rem;
            height: 2.8rem;
            border-radius: 10px;
            background: var(--gradient-primary);
            color: var(--primary-text) !important;
            font-weight: 900;
            font-size: 0.95rem;
            letter-spacing: -0.5px;
            box-shadow: var(--glow);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .brand-logo:hover {{
            transform: scale(1.05);
            box-shadow: var(--glow), 0 0 40px rgba(20,184,166,0.12);
        }}

        .brand-title {{
            margin: 0;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            line-height: 1.15;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .brand-subtitle {{
            margin: 0.15rem 0 0;
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-secondary) !important;
            letter-spacing: 0.3px;
        }}

        /* ── Section Titles ────────────────────────────────────── */
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            margin: 1.2rem 0 0.3rem;
            letter-spacing: -0.3px;
            animation: fadeInUp 0.4s ease-out;
        }}

        .section-copy {{
            margin: 0 0 0.6rem;
            font-size: 0.88rem;
            color: var(--text-secondary) !important;
            line-height: 1.5;
        }}

        /* ── Glass Cards ──────────────────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            background: var(--surface) !important;
            backdrop-filter: blur(16px) saturate(1.4);
            -webkit-backdrop-filter: blur(16px) saturate(1.4);
            box-shadow: var(--card-shadow);
            transition: border-color 0.25s ease, box-shadow 0.35s ease, transform 0.25s ease;
            animation: fadeInUp 0.45s ease-out;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: var(--border-hover) !important;
            box-shadow: var(--card-shadow-hover);
            transform: translateY(-1px);
        }}

        /* ── Runtime Chip ─────────────────────────────────────── */
        .runtime-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--surface);
            backdrop-filter: blur(8px);
            font-size: 0.74rem;
            font-weight: 600;
            color: var(--text-secondary) !important;
            transition: border-color 0.2s ease;
        }}
        .runtime-chip:hover {{
            border-color: var(--success);
        }}

        .runtime-dot {{
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: var(--success);
            animation: pulse-dot 2s ease-in-out infinite;
        }}

        /* ── Image Panels ─────────────────────────────────────── */
        .image-heading {{
            margin: 0 0 0.5rem;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-secondary) !important;
        }}

        [data-testid="stImage"] img {{
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: var(--image-well);
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        [data-testid="stImage"] img:hover {{
            border-color: var(--primary);
            box-shadow: var(--glow);
        }}

        /* ── File Uploader ────────────────────────────────────── */
        [data-testid="stFileUploader"] {{
            border: 1.5px dashed var(--border) !important;
            border-radius: 12px;
            background: var(--surface) !important;
            backdrop-filter: blur(8px);
            padding: 0.8rem;
            transition: border-color 0.3s ease, background 0.3s ease;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: var(--primary) !important;
            background: var(--surface-hover) !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
            border: 0 !important;
        }}

        /* ── Buttons ──────────────────────────────────────────── */
        .stButton > button,
        .stDownloadButton > button {{
            min-height: 2.7rem;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: var(--surface) !important;
            backdrop-filter: blur(8px);
            color: var(--text) !important;
            font-weight: 650;
            font-size: 0.85rem;
            letter-spacing: 0.2px;
            transition: all 0.25s ease;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: var(--primary);
            background: var(--surface-hover) !important;
            box-shadow: var(--glow);
            transform: translateY(-1px);
        }}

        /* Primary button — gradient */
        .stButton > button[kind="primary"] {{
            border: none;
            background: var(--gradient-primary) !important;
            background-size: 200% 200%;
            animation: gradientShift 4s ease infinite;
            color: var(--primary-text) !important;
            box-shadow: var(--glow);
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: var(--glow), 0 4px 20px rgba(20,184,166,0.25);
            transform: translateY(-2px);
        }}

        /* ── Metrics ──────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem;
            background: var(--surface) !important;
            backdrop-filter: blur(12px);
            box-shadow: var(--card-shadow);
            transition: border-color 0.25s ease, box-shadow 0.3s ease, transform 0.25s ease;
            animation: fadeInUp 0.5s ease-out;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: var(--border-hover);
            box-shadow: var(--card-shadow-hover);
            transform: translateY(-2px);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
            color: var(--text-secondary) !important;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
            color: var(--text) !important;
            font-weight: 700;
        }}

        /* ── Alerts ───────────────────────────────────────────── */
        [data-testid="stAlert"] {{
            border: 1px solid var(--border) !important;
            border-radius: 10px;
            background: var(--surface) !important;
            backdrop-filter: blur(8px);
        }}

        /* ── Dividers ─────────────────────────────────────────── */
        hr {{
            border-color: var(--border) !important;
            opacity: 0.6;
        }}

        /* ── Footer ───────────────────────────────────────────── */
        .app-footer {{
            text-align: center;
            padding: 1rem 0 0.5rem;
            animation: fadeInUp 0.6s ease-out;
        }}
        .app-footer p {{
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-secondary) !important;
            letter-spacing: 0.3px;
        }}
        .app-footer .footer-brand {{
            font-weight: 700;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Toggle overrides ─────────────────────────────────── */
        [data-testid="stToggle"] label span {{
            color: var(--text) !important;
        }}

        /* ── Captions ─────────────────────────────────────────── */
        [data-testid="stCaptionContainer"] p {{
            color: var(--text-secondary) !important;
        }}

        /* ── Responsive ───────────────────────────────────────── */
        @media (max-width: 768px) {{
            .block-container {{
                padding: 0.8rem 0.8rem 1.5rem;
            }}
            .brand-title {{
                font-size: 1.2rem;
            }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_uploaded_image(
    uploaded_file,
) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    """Validate an upload and keep its original pixel precision for inference."""
    declared_size = int(getattr(uploaded_file, "size", 0))
    if declared_size > MAX_UPLOAD_BYTES:
        return None, None, "The uploaded file is larger than the 20 MB limit."

    raw_bytes = uploaded_file.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return None, None, "The uploaded file is larger than the 20 MB limit."

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()
            if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                return (
                    None,
                    None,
                    f"Image dimensions must not exceed {MAX_IMAGE_DIMENSION} pixels on either side.",
                )
            if image.width == 0 or image.height == 0:
                return None, None, "The uploaded image has invalid dimensions."
            signature = hashlib.sha256(raw_bytes).hexdigest()
            return image.copy(), signature, None
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return None, None, "The file could not be read as a valid PNG, JPG, or TIFF image."


def png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_resource(show_spinner="Preparing the Pix2Pix inference engine…")
def get_engine(checkpoint_path: str) -> LandsatColorizationInference:
    """Cache the inference engine so the model is loaded only once."""
    return LandsatColorizationInference(checkpoint_path=checkpoint_path)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "light_mode" not in st.session_state:
    st.session_state["light_mode"] = False
if "upload_generation" not in st.session_state:
    st.session_state["upload_generation"] = 0

# ---------------------------------------------------------------------------
# Theme injection
# ---------------------------------------------------------------------------
inject_css(get_theme(st.session_state["light_mode"]))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_left, header_right = st.columns([0.82, 0.18])
with header_left:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-logo">IN</div>
            <div>
                <p class="brand-title">InfraNova AI</p>
                <p class="brand-subtitle">Landsat 9 thermal infrared → RGB synthesis</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.toggle("Light mode", key="light_mode")

# ---------------------------------------------------------------------------
# Engine loading — graceful fallback
# ---------------------------------------------------------------------------
engine: Optional[LandsatColorizationInference] = None
engine_error: Optional[str] = None

try:
    engine = get_engine(str(CHECKPOINT_PATH))
except FileNotFoundError:
    engine_error = f"Model checkpoint not found at `{CHECKPOINT_PATH}`. Please place the trained `.pth` file there."
    logger.error(engine_error)
except Exception:
    engine_error = "Failed to load the inference engine. Check the console for details."
    logger.error("Engine loading failed:\n%s", traceback.format_exc())

if engine is not None:
    runtime = "GPU acceleration" if engine.device.type == "cuda" else "CPU execution"
else:
    runtime = "unavailable"

# ---------------------------------------------------------------------------
# Workspace section
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">Thermal analysis workspace</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Upload a single Landsat 9 Band 10 image, then generate an RGB-like visual interpretation.</p>',
    unsafe_allow_html=True,
)

# Controls row
ctrl_tta, ctrl_enhance, ctrl_runtime = st.columns([0.26, 0.26, 0.48])
with ctrl_tta:
    use_tta = st.toggle(
        "Test-time augmentation",
        value=False,
        help="Averages four geometric predictions. Improves consistency but takes longer.",
    )
with ctrl_enhance:
    use_enhance = st.toggle(
        "Auto contrast",
        value=False,
        help="Applies CLAHE contrast enhancement to the generated RGB image.",
    )
with ctrl_runtime:
    st.markdown(
        f'<span class="runtime-chip"><span class="runtime-dot"></span>Pix2Pix · {runtime}</span>',
        unsafe_allow_html=True,
    )

# Show engine error banner if needed
if engine_error is not None:
    st.error(engine_error)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a thermal infrared image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    key=f"thermal_upload_{st.session_state['upload_generation']}",
    label_visibility="collapsed",
)
st.caption("PNG, JPG, or TIFF · Maximum 20 MB · Maximum 4096 × 4096 px")

upload_error = None
if uploaded_file is not None:
    uploaded_image, signature, upload_error = load_uploaded_image(uploaded_file)
    if upload_error is None and signature != st.session_state.get("input_signature"):
        st.session_state["input_image"] = uploaded_image
        st.session_state["input_name"] = uploaded_file.name
        st.session_state["input_signature"] = signature
        for key in ("output_image", "inference_time", "used_tta", "used_enhance"):
            st.session_state.pop(key, None)

if upload_error:
    st.error(upload_error)

input_image = st.session_state.get("input_image")
output_image = st.session_state.get("output_image")

# ---------------------------------------------------------------------------
# Image display — side by side in glass cards
# ---------------------------------------------------------------------------
st.markdown("---")
input_col, output_col = st.columns(2, gap="large")

with input_col:
    with st.container(border=True):
        st.markdown(
            '<p class="image-heading">Model input · thermal infrared</p>',
            unsafe_allow_html=True,
        )
        if input_image is None:
            st.info("Upload a thermal image to start a new analysis.")
        else:
            thermal_preview = visualize_tir_as_thermal(input_image).resize(
                (PREVIEW_SIZE, PREVIEW_SIZE),
                RESAMPLING.BICUBIC,
            )
            st.image(thermal_preview, use_container_width=True)
            st.caption(
                f"{st.session_state.get('input_name', 'Uploaded image')} · "
                f"{input_image.width} × {input_image.height} · original precision preserved"
            )

with output_col:
    with st.container(border=True):
        st.markdown(
            '<p class="image-heading">Generated RGB interpretation</p>',
            unsafe_allow_html=True,
        )
        if output_image is None:
            st.info("The generated RGB output will appear here.")
        else:
            st.image(output_image, use_container_width=True)
            st.caption("Synthesised from the current thermal input.")

# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
st.markdown("---")
action_col, download_col, clear_col = st.columns([0.38, 0.38, 0.24])

with action_col:
    process_button = st.button(
        "✦  Generate RGB",
        type="primary",
        use_container_width=True,
        disabled=(input_image is None or engine is None),
    )
with download_col:
    if output_image is None:
        st.button("Download PNG", use_container_width=True, disabled=True)
    else:
        st.download_button(
            "↓  Download PNG",
            data=png_bytes(output_image),
            file_name="infranova_rgb_interpretation.png",
            mime="image/png",
            use_container_width=True,
        )
with clear_col:
    clear_button = st.button("Clear workspace", use_container_width=True)

# ---------------------------------------------------------------------------
# Clear workspace
# ---------------------------------------------------------------------------
if clear_button:
    for key in (
        "input_image",
        "input_name",
        "input_signature",
        "output_image",
        "inference_time",
        "used_tta",
        "used_enhance",
    ):
        st.session_state.pop(key, None)
    st.session_state["upload_generation"] += 1
    st.rerun()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
if process_button and input_image is not None and engine is not None:
    try:
        with st.spinner("Generating an RGB interpretation from the thermal input…"):
            start_time = time.perf_counter()
            prediction = engine.predict(input_image, use_tta=use_tta)
            generated_image = prediction["rgb"]
            if use_enhance:
                generated_image = enhance_output(generated_image)

        st.session_state["output_image"] = generated_image
        st.session_state["inference_time"] = time.perf_counter() - start_time
        st.session_state["used_tta"] = use_tta
        st.session_state["used_enhance"] = use_enhance
        st.rerun()
    except FileNotFoundError:
        st.error(f"Model checkpoint not found: `{CHECKPOINT_PATH}`")
        logger.error("Checkpoint missing during inference:\n%s", traceback.format_exc())
    except Exception:
        st.error(
            "Generation failed. Check that the checkpoint and image format are valid, then try again."
        )
        logger.error("Inference failed:\n%s", traceback.format_exc())

# ---------------------------------------------------------------------------
# Result metrics
# ---------------------------------------------------------------------------
if output_image is not None:
    st.markdown("---")
    st.markdown('<p class="section-title">Current result</p>', unsafe_allow_html=True)

    m_time, m_input, m_output, m_mode = st.columns(4)
    with m_time:
        st.metric("Inference time", f"{st.session_state.get('inference_time', 0.0):.2f} s")
    with m_input:
        if input_image is not None:
            st.metric("Input size", f"{input_image.width} × {input_image.height}")
        else:
            st.metric("Input size", "—")
    with m_output:
        st.metric("Output size", f"{output_image.width} × {output_image.height}")
    with m_mode:
        mode = "TTA ×4" if st.session_state.get("used_tta") else "Single pass"
        st.metric("Inference mode", mode)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="app-footer">
        <p>
            Thermal infrared does not contain true visible colour.
            The output is a learned, plausible RGB-like interpretation
            and should not be treated as ground truth.
        </p>
        <p style="margin-top: 0.5rem;">
            <span class="footer-brand">InfraNova AI</span>
            &nbsp;·&nbsp; Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp; ISRO
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
