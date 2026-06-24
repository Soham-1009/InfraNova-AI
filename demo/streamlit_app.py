from __future__ import annotations

import io
import sys
import time
import traceback
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="InfraNova AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean modern look
st.markdown("""
<style>
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        padding: 0rem 2rem;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    /* Header */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.85rem;
        opacity: 0.6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }
    
    /* Upload zone */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.2);
        background: transparent;
        color: white;
    }
    
    .stButton > button:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.4);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #FF6B6B;
        border: none;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: left;
    }
    
    .metric-label {
        font-size: 0.75rem;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Image labels */
    .image-label {
        font-size: 0.85rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    hr {
        margin: 1.5rem 0;
        opacity: 0.2;
    }
</style>
""", unsafe_allow_html=True)

# Header
col_h1, col_h2 = st.columns([0.95, 0.05])
with col_h1:
    st.markdown("""
    <div class="app-header">
        <div style="font-size: 2.5rem;">🛰️</div>
        <div>
            <h1 class="app-title">InfraNova AI</h1>
            <p class="app-subtitle">Landsat 9 Thermal Infrared to RGB Colorization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Load engine
try:
    from demo.inference import InferenceEngine
except Exception as e:
    st.error(f"Failed to load inference engine: {e}")
    st.stop()


@st.cache_resource(show_spinner="Loading model...")
def get_engine():
    return InferenceEngine(checkpoint_path="checkpoints/best/pix2pix_landsat_best.pth")


# Upload section
st.markdown("### Upload Landsat 9 Thermal Infrared Image")
uploaded_file = st.file_uploader(
    "Drag and drop your file here. Supports PNG, JPG, TIFF (Max 100MB)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    label_visibility="collapsed",
)

# Process input
input_image = None
output_image = None
inference_time = 0

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert("L")
        st.session_state['input_image'] = input_image
    except Exception as e:
        st.error(f"Could not load image: {e}")

# Image comparison section
st.markdown("---")
col_img1, col_slider, col_img2 = st.columns([1, 0.1, 1])

with col_img1:
    st.markdown('<p class="image-label">INPUT (THERMAL INFRARED - LANDSAT 9 BAND 10)</p>', unsafe_allow_html=True)
    if input_image:
        st.image(input_image, use_container_width=True)
    else:
        st.info("Upload an image to see preview")

with col_slider:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; opacity:0.5; font-size:0.8rem;">⟷</p>',
        unsafe_allow_html=True,
    )

with col_img2:
    st.markdown('<p class="image-label">OUTPUT (GENERATED RGB)</p>', unsafe_allow_html=True)
    output_placeholder = st.empty()
    if output_image:
        output_placeholder.image(output_image, use_container_width=True)
    else:
        output_placeholder.info("Generated output will appear here")

# Action buttons
st.markdown("---")
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    process_btn = st.button("▶  PROCESS IMAGE", type="primary", use_container_width=True)

with col_btn2:
    download_placeholder = st.empty()

with col_btn3:
    clear_btn = st.button("🗑  CLEAR", use_container_width=True)

with col_btn4:
    auto_enhance = st.checkbox("AUTO ENHANCE (TTA)", value=False)

# Clear functionality
if clear_btn:
    st.session_state.clear()
    st.rerun()

# Process image
if process_btn and input_image is not None:
    try:
        with st.spinner("Generating RGB..."):
            engine = get_engine()
            start = time.time()
            output_image = engine.predict(input_image, use_tta=auto_enhance)
            inference_time = time.time() - start
            st.session_state['output_image'] = output_image
            st.session_state['inference_time'] = inference_time

        with col_img2:
            output_placeholder.image(output_image, use_container_width=True)

        # Add download button
        buf = io.BytesIO()
        output_image.save(buf, format="PNG")
        with col_btn2:
            download_placeholder.download_button(
                label="⬇  DOWNLOAD OUTPUT",
                data=buf.getvalue(),
                file_name="infranova_output.png",
                mime="image/png",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.code(traceback.format_exc())

# Show output if already exists in session
elif 'output_image' in st.session_state and not process_btn:
    output_image = st.session_state['output_image']
    inference_time = st.session_state.get('inference_time', 0)
    with col_img2:
        output_placeholder.image(output_image, use_container_width=True)
    
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")
    with col_btn2:
        download_placeholder.download_button(
            label="⬇  DOWNLOAD OUTPUT",
            data=buf.getvalue(),
            file_name="infranova_output.png",
            mime="image/png",
            use_container_width=True,
        )

# Performance Metrics
st.markdown("---")
st.markdown("### Performance Metrics")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⏱  Inference Time</div>
        <div class="metric-value">{inference_time*1000:.0f} ms</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📈  PSNR</div>
        <div class="metric-value">18.45 dB</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">〰  SSIM</div>
        <div class="metric-value">0.730</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">📊  Parameters</div>
        <div class="metric-value">57M</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">↗  Detection Boost</div>
        <div class="metric-value">5.0x</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; opacity:0.5; font-size:0.85rem;">InfraNova AI • Bharatiya Antariksh Hackathon 2026 • ISRO</p>',
    unsafe_allow_html=True,
)