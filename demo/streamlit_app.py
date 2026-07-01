from __future__ import annotations

import io
import sys
import time
import traceback
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="InfraNova AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main { padding: 0rem 2rem; }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 1400px;
    }
    
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    
    .app-title { font-size: 2rem; font-weight: 700; margin: 0; }
    
    .app-subtitle {
        font-size: 0.85rem;
        opacity: 0.6;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0;
    }
    
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
    }
    
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
    
    .metric-value { font-size: 1.5rem; font-weight: 600; }
    
    .image-label {
        font-size: 0.85rem;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    hr { margin: 1.5rem 0; opacity: 0.2; }
    
    .info-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
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


@st.cache_resource(show_spinner="Loading Pix2Pix model (first time only)...")
def get_engine():
    return InferenceEngine(checkpoint_path="checkpoints/best/pix2pix_landsat_best.pth")


# Sidebar
with st.sidebar:
    st.header("About InfraNova AI")
    st.write("ISRO BAH 2026")
    st.write("Thermal Infrared to RGB Translation")
    
    st.subheader("Model")
    st.write("Pix2Pix Conditional GAN")
    st.write("57M parameters")
    st.write("Trained on 276 global regions")
    st.write("9,936 paired patches")
    
    st.subheader("Architecture")
    st.write("- U-Net Generator (54M params)")
    st.write("- PatchGAN Discriminator (2.8M params)")
    st.write("- Combined L1 + Adversarial + Perceptual + SSIM loss")
    
    st.subheader("Performance")
    st.write("- SSIM: 0.695")
    st.write("- PSNR: 18.52 dB")
    st.write("- 250 training epochs")
    
    st.subheader("Options")
    use_tta = st.checkbox("Enable TTA (better quality, 4x slower)", value=False)
    use_enhance = st.checkbox("Auto Enhance (CLAHE)", value=False)
    
    st.divider()
    st.write("Team: InfraNova AI")
    st.write("Hackathon: Bharatiya Antariksh Hackathon 2026")

# Sample images
sample_dir = Path("data/landsat9/splits/test")
if sample_dir.exists():
    import numpy as np
    test_samples = sorted(list(sample_dir.iterdir()))[:4]
    
    if test_samples:
        st.markdown("### Quick Test with Sample Images")
        cols = st.columns(len(test_samples))
        
        for i, (col, sample) in enumerate(zip(cols, test_samples)):
            with col:
                tir = np.load(sample / 'tir_100m.npy')
                tir_display = ((tir - tir.min()) / (tir.max() - tir.min() + 1e-8) * 255).astype(np.uint8)
                st.image(tir_display, caption=f"Sample {i+1}", use_container_width=True)
                if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                    st.session_state['input_image'] = Image.fromarray(tir_display)
                    st.session_state['from_sample'] = True

# Upload section
st.markdown("---")
st.markdown("### Upload Landsat 9 Thermal Infrared Image")
uploaded_file = st.file_uploader(
    "Supports PNG, JPG, TIFF (Max 100MB)",
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
        st.session_state['from_sample'] = False
    except Exception as e:
        st.error(f"Could not load image: {e}")
elif 'input_image' in st.session_state:
    input_image = st.session_state['input_image']

# Image display
st.markdown("---")
col_img1, col_mid, col_img2 = st.columns([1, 0.1, 1])

with col_img1:
    st.markdown('<p class="image-label">INPUT (THERMAL INFRARED - LANDSAT 9 BAND 10)</p>', unsafe_allow_html=True)
    if input_image:
        source = "Sample" if st.session_state.get('from_sample', False) else "Uploaded"
        st.image(input_image, caption=f"IR Input ({source})", use_container_width=True)
    else:
        st.info("Upload an image or select a sample above")

with col_mid:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; opacity:0.5; font-size:1.5rem;">→</p>', unsafe_allow_html=True)

with col_img2:
    st.markdown('<p class="image-label">OUTPUT (GENERATED RGB)</p>', unsafe_allow_html=True)
    output_placeholder = st.empty()
    if 'output_image' in st.session_state:
        output_placeholder.image(st.session_state['output_image'], caption="Generated RGB", use_container_width=True)
    else:
        output_placeholder.info("Generated output will appear here")

# Action buttons
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    process_btn = st.button("▶  PROCESS IMAGE", type="primary", use_container_width=True)

with col_btn2:
    download_placeholder = st.empty()

with col_btn3:
    clear_btn = st.button("🗑  CLEAR", use_container_width=True)

# Clear
if clear_btn:
    st.session_state.clear()
    st.rerun()

# Process
if process_btn and input_image is not None:
    try:
        with st.spinner("Loading model and generating RGB..."):
            engine = get_engine()
            start = time.time()
            output_image = engine.predict(input_image, use_tta=use_tta)
            inference_time = time.time() - start
            
            # Apply CLAHE enhancement if enabled
            if use_enhance:
                from demo.utils import enhance_output
                output_image = enhance_output(output_image)
            
            st.session_state['output_image'] = output_image
            st.session_state['inference_time'] = inference_time

        with col_img2:
            output_placeholder.image(output_image, caption="Generated RGB", use_container_width=True)

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

elif 'output_image' in st.session_state:
    output_image = st.session_state['output_image']
    inference_time = st.session_state.get('inference_time', 0)
    
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
    display_time = st.session_state.get('inference_time', 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">⏱  Inference Time</div>
        <div class="metric-value">{display_time*1000:.0f} ms</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📈  PSNR</div>
        <div class="metric-value">18.52 dB</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">〰  SSIM</div>
        <div class="metric-value">0.695</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">📊  Parameters</div>
        <div class="metric-value">57M</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🌍  Regions</div>
        <div class="metric-value">250</div>
    </div>
    """, unsafe_allow_html=True)

# Technical Details
st.markdown("---")
st.markdown("### Technical Details")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("""
    <div class="info-box">
        <h4>Training Data</h4>
        <ul>
            <li>Satellite: Landsat 9 (NASA/USGS)</li>
            <li>Input: Band 10 Thermal Infrared (10.9 μm)</li>
            <li>Output: Bands 2,3,4 (Blue, Green, Red)</li>
            <li>Regions: 100 Indian + 90 International + 60 Landscapes</li>
            <li>Patches: 9,936 paired samples</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("""
    <div class="info-box">
        <h4>Architecture</h4>
        <ul>
            <li>Model: Pix2Pix Conditional GAN</li>
            <li>Generator: U-Net with 8 encoder-decoder blocks</li>
            <li>Discriminator: 70x70 PatchGAN with Spectral Norm</li>
            <li>Loss: L1 + Adversarial + VGG Perceptual + SSIM</li>
            <li>Training: 250 epochs with label smoothing + noise injection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align:center; opacity:0.5; font-size:0.85rem;">'
    'InfraNova AI • Bharatiya Antariksh Hackathon 2026 • ISRO'
    '</p>',
    unsafe_allow_html=True,
)