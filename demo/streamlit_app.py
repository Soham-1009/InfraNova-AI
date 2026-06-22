from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.inference import InferenceEngine
from demo.utils import save_output, load_sample_images

st.set_page_config(
    page_title="InfraNova AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            font-weight: 600;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

CHECKPOINT_PATH = "checkpoints/best/pix2pix_best.pth"


@st.cache_resource(show_spinner="Loading Pix2Pix model...")
def get_engine(checkpoint_path: str) -> InferenceEngine:
    """Cached model loader for Streamlit."""
    return InferenceEngine(checkpoint_path=checkpoint_path)


def read_image_from_upload(uploaded_file) -> Image.Image:
    """Safely read a Streamlit uploaded file as PIL grayscale image."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    return Image.open(uploaded_file).convert("L")


def main() -> None:
    st.title("InfraNova AI")
    st.caption("Infrared Image Colorization and Enhancement for Improved Object Interpretation")

    with st.sidebar:
        st.header("Project Info")
        st.write("ISRO BAH 2026")
        st.write("Infrared to RGB satellite translation")
        st.write("Model: Pix2Pix Conditional GAN")
        st.write("Input: Grayscale IR image")
        st.write("Output: 256x256 RGB image")

        st.subheader("Architecture")
        st.write(
            "- U-Net Generator (54M params)\n"
            "- PatchGAN Discriminator (2.8M params)\n"
            "- Combined L1 + Adversarial + Perceptual + SSIM loss\n"
            "- 57M total parameters"
        )

        st.subheader("Options")
        tta_enabled = st.toggle("Enable Test-Time Augmentation", value=False, help="Improves quality, 4x slower")
        
        st.divider()
        st.write("Team: InfraNova AI")
        st.write("Hackathon: Bharatiya Antariksh Hackathon 2026")
        st.write("Organizer: ISRO")

    # Sample images section
    sample_images = load_sample_images("data/samples")
    
    if sample_images:
        st.subheader("Try Sample Images")
        sample_cols = st.columns(min(len(sample_images), 4))
        selected_sample = None
        
        for i, (col, img) in enumerate(zip(sample_cols, sample_images[:4])):
            with col:
                st.image(img, caption=f"Sample {i+1}", use_container_width=True)
                if st.button(f"Use Sample {i+1}", key=f"sample_{i}"):
                    selected_sample = img
        
        if selected_sample is not None:
            st.session_state['input_image'] = selected_sample
            st.session_state['from_sample'] = True

    st.divider()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload IR Image")
        uploaded_file = st.file_uploader(
            "Accepted formats: PNG, JPG, TIFF",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=False,
        )

        run_button = st.button("Generate RGB", type="primary")

        # Determine input source
        input_image = None
        if uploaded_file is not None:
            try:
                input_image = read_image_from_upload(uploaded_file)
                st.session_state['input_image'] = input_image
                st.session_state['from_sample'] = False
            except Exception as exc:
                st.error(f"Failed to read image: {exc}")
                return
        elif 'input_image' in st.session_state:
            input_image = st.session_state['input_image']

        if input_image is not None:
            source = "Sample Image" if st.session_state.get('from_sample', False) else "Uploaded Image"
            st.image(input_image, caption=f"IR Input ({source})", use_container_width=True)
        else:
            st.info("Upload a grayscale IR image or select a sample above to begin.")

    with col2:
        st.subheader("Generated Output")
        output_placeholder = st.empty()
        metrics_placeholder = st.empty()

    if run_button:
        if input_image is None:
            st.warning("Please upload an image or select a sample first.")
            return

        try:
            engine = get_engine(CHECKPOINT_PATH)

            start = time.perf_counter()
            output_image = engine.predict(input_image, use_tta=tta_enabled)
            inference_time = time.perf_counter() - start

            with col2:
                output_placeholder.image(output_image, caption="Generated RGB Output", use_container_width=True)

                buf = io.BytesIO()
                output_image.save(buf, format="PNG")
                st.download_button(
                    label="Download Result",
                    data=buf.getvalue(),
                    file_name="infranova_output.png",
                    mime="image/png",
                    use_container_width=True,
                )

                metrics_placeholder.markdown(
                    f"""
                    <div class="card">
                        <b>Inference Time:</b> {inference_time:.3f} s<br>
                        <b>Output Dimensions:</b> {output_image.size[0]} x {output_image.size[1]}<br>
                        <b>Device:</b> {engine.device.type.upper()}<br>
                        <b>TTA:</b> {"Enabled" if tta_enabled else "Disabled"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Comparison slider below
            st.subheader("Interactive Comparison")
            image_comparison(
                img1=input_image.convert("RGB"),
                img2=output_image,
                label1="IR Input",
                label2="Generated RGB",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
            )

            save_output(output_image, "outputs/visualizations/streamlit_latest.png")

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; opacity:0.8;">
            InfraNova AI | BAH 2026 | Team InfraNova AI
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()