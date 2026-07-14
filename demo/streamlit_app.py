from __future__ import annotations

import base64
import hashlib
import io
import logging
import sys
import time
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
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
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
def get_theme() -> Dict[str, str]:
    """Return a colour palette for the requested appearance."""
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
    bg = theme["bg"]
    bg_gradient = theme["bg_gradient"]
    surface = theme["surface"]
    surface_solid = theme["surface_solid"]
    surface_hover = theme["surface_hover"]
    text = theme["text"]
    text_secondary = theme["text_secondary"]
    border = theme["border"]
    border_hover = theme["border_hover"]
    primary = theme["primary"]
    primary_hover = theme["primary_hover"]
    primary_text = theme["primary_text"]
    accent = theme["accent"]
    success = theme["success"]
    image_well = theme["image_well"]
    card_shadow = theme["card_shadow"]
    card_shadow_hover = theme["card_shadow_hover"]
    glow = theme["glow"]
    gradient_primary = theme["gradient_primary"]
    gradient_accent = theme["gradient_accent"]

    css = f"""
    <style data-theme-id="{bg}">
        /* ── Google Fonts ─────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        /* ── Reset & Base ─────────────────────────────────────── */
        html, body, .stApp {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}

        #MainMenu, footer,
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        [data-testid="stAppViewContainer"], .stApp {{
            background: {bg_gradient} !important;
            color: {text} !important;
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
            color: {text} !important;
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
            width: 3.2rem;
            height: 3.2rem;
            border-radius: 8px;
            object-fit: contain;
            background: #ffffff;
            box-shadow: {glow};
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .brand-logo:hover {{
            transform: scale(1.05);
            box-shadow: {glow}, 0 0 40px rgba(20,184,166,0.12);
        }}

        .brand-title {{
            margin: 0;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            line-height: 1.15;
            background: {gradient_primary};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .brand-subtitle {{
            margin: 0.15rem 0 0;
            font-size: 0.78rem;
            font-weight: 500;
            color: {text_secondary} !important;
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
            color: {text_secondary} !important;
            line-height: 1.5;
        }}

        /* ── Glass Cards ──────────────────────────────────────── */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            background: {surface} !important;
            backdrop-filter: blur(16px) saturate(1.4);
            -webkit-backdrop-filter: blur(16px) saturate(1.4);
            box-shadow: {card_shadow};
            transition: border-color 0.25s ease, box-shadow 0.35s ease, transform 0.25s ease;
            animation: fadeInUp 0.45s ease-out;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: {border_hover} !important;
            box-shadow: {card_shadow_hover};
            transform: translateY(-1px);
        }}

        /* ── Runtime Chip ─────────────────────────────────────── */
        .runtime-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid {border};
            background: {surface};
            backdrop-filter: blur(8px);
            font-size: 0.74rem;
            font-weight: 600;
            color: {text_secondary} !important;
            transition: border-color 0.2s ease;
        }}
        .runtime-chip:hover {{
            border-color: {success};
        }}

        .runtime-dot {{
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: {success};
            animation: pulse-dot 2s ease-in-out infinite;
        }}

        /* ── Image Panels ─────────────────────────────────────── */
        .image-heading {{
            margin: 0 0 0.5rem;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: {text_secondary} !important;
        }}

        [data-testid="stImage"] img {{
            width: 100%;
            aspect-ratio: 1 / 1;
            object-fit: cover;
            border: 1px solid {border};
            border-radius: 10px;
            background: {image_well};
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        [data-testid="stImage"] img:hover {{
            border-color: {primary};
            box-shadow: {glow};
        }}

        /* ── File Uploader ────────────────────────────────────── */
        [data-testid="stFileUploader"] {{
            border: 1.5px dashed {border} !important;
            border-radius: 12px;
            background: {surface} !important;
            backdrop-filter: blur(8px);
            padding: 0.8rem;
            transition: border-color 0.3s ease, background 0.3s ease;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color: {primary} !important;
            background: {surface_hover} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
            border: 0 !important;
            min-height: 90px !important;
            display: flex !important;
            align-items: center !important;
        }}
        [data-testid="stFileUploader"] button {{
            background: {surface_solid} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}
        [data-testid="stFileUploader"] button:hover {{
            background: {surface_hover} !important;
            border-color: {primary} !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span {{
            color: {text} !important;
        }}

        /* ── Buttons ──────────────────────────────────────────── */
        .stButton > button,
        .stDownloadButton > button {{
            min-height: 2.7rem;
            border-radius: 10px;
            border: 1px solid {border};
            background: {surface} !important;
            backdrop-filter: blur(8px);
            color: {text} !important;
            font-weight: 650;
            font-size: 0.85rem;
            letter-spacing: 0.2px;
            transition: all 0.25s ease;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover {{
            border-color: {primary};
            background: {surface_hover} !important;
            box-shadow: {glow};
            transform: translateY(-1px);
        }}

        /* Primary button — gradient */
        .stButton > button[kind="primary"] {{
            border: none;
            background: {gradient_primary} !important;
            background-size: 200% 200%;
            animation: gradientShift 4s ease infinite;
            color: {primary_text} !important;
            box-shadow: {glow};
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: {glow}, 0 4px 20px rgba(20,184,166,0.25);
            transform: translateY(-2px);
        }}

        /* ── Metrics ──────────────────────────────────────────── */
        [data-testid="stMetric"] {{
            border: 1px solid {border};
            border-radius: 12px;
            padding: 0.85rem;
            background: {surface} !important;
            backdrop-filter: blur(12px);
            box-shadow: {card_shadow};
            transition: border-color 0.25s ease, box-shadow 0.3s ease, transform 0.25s ease;
            animation: fadeInUp 0.5s ease-out;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {border_hover};
            box-shadow: {card_shadow_hover};
            transform: translateY(-2px);
        }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
            color: {text_secondary} !important;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
            color: {text} !important;
            font-weight: 700;
        }}

        /* ── Alerts ───────────────────────────────────────────── */
        [data-testid="stAlert"] {{
            border: 1px solid {border} !important;
            border-radius: 10px;
            background: {surface} !important;
            backdrop-filter: blur(8px);
        }}

        /* ── Dividers ─────────────────────────────────────────── */
        hr {{
            border-color: {border} !important;
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
            color: {text_secondary} !important;
            letter-spacing: 0.3px;
        }}
        .app-footer .footer-brand {{
            font-weight: 700;
            background: {gradient_primary};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Toggle overrides ─────────────────────────────────── */
        [data-testid="stToggle"] {{
            pointer-events: auto !important;
        }}
        [data-testid="stToggle"] label {{
            cursor: pointer !important;
        }}
        [data-testid="stToggle"] label span {{
            color: {text} !important;
        }}

        /* ── Captions ─────────────────────────────────────────── */
        [data-testid="stCaptionContainer"] p {{
            color: {text_secondary} !important;
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
        return None, None, "The uploaded file is larger than the 200 MB limit."

    raw_bytes = uploaded_file.getvalue()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return None, None, "The uploaded file is larger than the 200 MB limit."

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.load()

            # Fix: Automatically convert standard RGB uploads to Grayscale
            if image.mode in ("RGB", "RGBA"):
                image = image.convert("L")

            if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                return (
                    None,
                    None,
                    f"Image dimensions must not exceed {MAX_IMAGE_DIMENSION} pixels on either side.",
                )
            if image.width == 0 or image.height == 0:
                return None, None, "The uploaded image has invalid dimensions."

            if not is_supported_thermal_mode(image):
                return None, None, (
                    "Please upload a single-band thermal infrared image (for example, grayscale TIFF)."
                )

            signature = hashlib.sha256(raw_bytes).hexdigest()
            return image.copy(), signature, None
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        return None, None, "The file could not be read as a valid PNG, JPG, or TIFF image."


def png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def is_supported_thermal_mode(image: Image.Image) -> bool:
    """Return True only for single-band image modes suitable for thermal inference."""
    return image.mode in {"1", "L", "I", "I;16", "I;16B", "I;16L", "F", "P"}


@st.cache_resource(show_spinner="Preparing the Pix2Pix inference engine…")
def get_engine(checkpoint_path: str, checkpoint_mtime: float) -> LandsatColorizationInference:
    """Cache the inference engine so the model is loaded only once."""
    _ = checkpoint_mtime
    return LandsatColorizationInference(checkpoint_path=checkpoint_path)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "upload_generation" not in st.session_state:
    st.session_state["upload_generation"] = 0

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
logo_path = PROJECT_ROOT / "demo" / "logo.jpg"
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/jpeg;base64,{logo_base64}" class="brand-logo" alt="InfraNova AI">'
else:
    logo_html = '<div class="brand-logo">IN</div>'

st.markdown(
    f"""
    <div class="brand-row">
        {logo_html}
        <div>
            <p class="brand-title">InfraNova AI</p>
            <p class="brand-subtitle">Landsat 9 thermal infrared → RGB synthesis</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

inject_css(get_theme())

# ---------------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------------
engine: Optional[LandsatColorizationInference] = None
engine_error: Optional[str] = None

if not CHECKPOINT_PATH.exists():
    engine_error = (
        f"Model checkpoint not found at `{CHECKPOINT_PATH}`. "
        "Please place the trained `.pth` file there."
    )
    logger.error(engine_error)
else:
    try:
        engine = get_engine(str(CHECKPOINT_PATH), CHECKPOINT_PATH.stat().st_mtime)
    except Exception:
        engine_error = "Failed to load the inference engine. Check the console for details."
        logger.exception("Engine loading failed")

if engine is not None:
    # Fix: Safer device check that doesn't crash if device format is slightly different
    device_repr = str(getattr(engine, "device", "cpu"))
    runtime = "GPU acceleration" if "cuda" in device_repr or "mps" in device_repr else "CPU execution"
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

upload_error = None
if uploaded_file is not None:
    uploaded_image, signature, upload_error = load_uploaded_image(uploaded_file)
    if upload_error is None:
        if signature != st.session_state.get("input_signature"):
            st.session_state["input_image"] = uploaded_image
            st.session_state["input_name"] = uploaded_file.name
            st.session_state["input_signature"] = signature
            for key in ("output_image", "inference_time", "used_tta", "used_enhance"):
                st.session_state.pop(key, None)
    else:
        # Clear stale state when a previously valid image is replaced by an invalid upload
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

if upload_error:
    st.error(upload_error)

input_image = st.session_state.get("input_image")
output_image = st.session_state.get("output_image")

# ---------------------------------------------------------------------------
# Image display
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
        download_stem = Path(st.session_state.get("input_name", "infranova_rgb_interpretation")).stem
        st.download_button(
            "↓  Download PNG",
            data=png_bytes(output_image),
            file_name=f"{download_stem}_rgb.png",
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
        logger.exception("Checkpoint missing during inference")
    except Exception:
        st.error(
            "Generation failed. Check that the checkpoint and image format are valid, then try again."
        )
        logger.exception("Inference failed")

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