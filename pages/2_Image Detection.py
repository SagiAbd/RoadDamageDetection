import os
import logging
from pathlib import Path
from typing import NamedTuple
import warnings
import sys

import cv2
import numpy as np
import streamlit as st

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*symlinks on Windows.*")

try:
    from ultralytics import YOLO
    from PIL import Image
    from io import BytesIO
    import torch
    from huggingface_hub import hf_hub_download
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.info("Please install: pip install ultralytics huggingface_hub torch pillow opencv-python")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent

# Logger setup
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

# HuggingFace model repository and file
MODEL_REPO = "Nothingger/RDD_YOLO_pretrained"
MODEL_FILENAME = "YOLOv11x_RDD_Trained.pt"  # Adjust if different

def safe_extract_value(tensor_value):
    """Safely extract scalar value from tensor/array"""
    try:
        if hasattr(tensor_value, 'item'):
            return tensor_value.item()
        elif hasattr(tensor_value, '__len__') and len(tensor_value) > 0:
            return tensor_value[0]
        else:
            return tensor_value
    except:
        return float(tensor_value) if tensor_value is not None else 0

# Load the model from HuggingFace
@st.cache_resource
def load_model():
    """Load and cache the YOLO model in memory only"""
    try:
        # Method 1: Download to temporary location (no persistent cache)
        with st.spinner("Downloading model from Hugging Face..."):
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILENAME,
                    cache_dir=temp_dir,  # Use temporary directory
                    local_files_only=False
                )
                
                with st.spinner("Loading YOLO model..."):
                    net = YOLO(model_path)
                    
        st.success("✅ Model loaded successfully from Hugging Face!")
        return net
        
    except Exception as e:
        st.warning(f"Primary loading failed: {str(e)}")
        
        # Method 2: Try direct loading (fallback)
        try:
            st.info("🔄 Trying alternative loading method...")
            with st.spinner("Loading model directly from URL..."):
                # Try loading directly from HuggingFace URL
                net = YOLO(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILENAME}")
            st.success("✅ Model loaded with direct URL method!")
            return net
            
        except Exception as e2:
            # Method 3: Last fallback - try repo name directly
            try:
                st.info("🔄 Trying repository name method...")
                with st.spinner("Loading model from repository..."):
                    net = YOLO(MODEL_REPO)
                st.success("✅ Model loaded with repository method!")
                return net
                
            except Exception as e3:
                st.error(f"❌ All loading methods failed:")
                st.error(f"HuggingFace Hub: {str(e)}")
                st.error(f"Direct URL: {str(e2)}")
                st.error(f"Repository: {str(e3)}")
                
                st.markdown("### Troubleshooting:")
                st.markdown("1. Check internet connection")
                st.markdown("2. Verify repository name and model file")
                st.markdown("3. Try running: `pip install --upgrade ultralytics huggingface_hub`")
                st.markdown("4. Check if the model file exists in the repository")
                st.stop()

# Load model
net = load_model()

# Class labels for road damage detection
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack", 
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Main UI
st.title("🛣️ Road Damage Detection")
st.markdown("Upload images to detect and classify road damage including cracks and potholes.")

# Model info in sidebar
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Repository:** `{MODEL_REPO}`")
    st.write(f"**Classes:** {len(CLASSES)}")
    for i, cls in enumerate(CLASSES):
        st.write(f"  {i}: {cls}")
    
    st.header("⚙️ Settings")
    score_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )

# Main content area
st.header("📤 Upload Images")
image_files = st.file_uploader(
    "Choose image files", 
    type=['png', 'jpg', 'jpeg'], 
    accept_multiple_files=True,
    help="Select one or more images to analyze"
)

if not image_files:
    st.info("👆 Please upload one or more images to get started")
    st.stop()

# Process uploaded images
st.header("🔍 Detection Results")

for idx, image_file in enumerate(image_files):
    with st.container():
        st.subheader(f"Image {idx + 1}: {image_file.name}")
        
        try:
            # Load and process image
            image = Image.open(image_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            _image = np.array(image)
            h_ori, w_ori = _image.shape[:2]

            # Prepare image for inference
            image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
            
            # Run inference
            with st.spinner(f"🔄 Analyzing image {idx + 1}..."):
                with torch.no_grad():  # Reduce memory usage
                    results = net.predict(
                        image_resized, 
                        conf=score_threshold, 
                        verbose=False,
                        save=False
                    )

            # Process detections safely
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.cpu().numpy()
                
                for _box in boxes:
                    try:
                        class_id = safe_extract_value(_box.cls)
                        confidence = safe_extract_value(_box.conf)
                        
                        if 0 <= class_id < len(CLASSES):
                            detection = Detection(
                                class_id=int(class_id),
                                label=CLASSES[int(class_id)],
                                score=float(confidence),
                                box=_box.xyxy[0].astype(int),
                            )
                            detections.append(detection)
                    except Exception as box_error:
                        logger.warning(f"Error processing detection box: {box_error}")
                        continue

            # Generate annotated image
            if results and len(results) > 0:
                try:
                    annotated_frame = results[0].plot()
                    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
                except:
                    _image_pred = _image.copy()
            else:
                _image_pred = _image.copy()

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Original Image**")
                st.image(_image, use_column_width=True)

            with col2:
                st.markdown("**Detection Results**")
                st.image(_image_pred, use_column_width=True)
                
                # Detection summary
                if detections:
                    st.success(f"Found {len(detections)} detection(s)")
                else:
                    st.info("No damage detected")

                # Download button
                try:
                    buffer = BytesIO()
                    Image.fromarray(_image_pred).save(buffer, format="PNG")
                    st.download_button(
                        label="💾 Download Result",
                        data=buffer.getvalue(),
                        file_name=f"{Path(image_file.name).stem}_detection.png",
                        mime="image/png",
                        key=f"download_{idx}",
                        help="Download the annotated image"
                    )
                except Exception as download_error:
                    st.warning(f"Download preparation failed: {download_error}")
                
        except Exception as e:
            st.error(f"❌ Error processing {image_file.name}: {str(e)}")
            continue
        
        if idx < len(image_files) - 1:  # Add separator except for last image
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips for Better Results")
st.markdown("""
- **Image Quality**: Use clear, well-lit images for best results
- **Confidence Threshold**: 
  - Higher (0.5-0.8): Fewer, more certain detections
  - Lower (0.1-0.3): More detections, may include false positives
- **Supported Formats**: PNG, JPG, JPEG
- **Multiple Images**: Upload several images at once for batch processing
""")

st.markdown("### 🔧 System Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}")
with col2:
    st.metric("PyTorch Version", torch.__version__)
with col3:
    st.metric("Classes", len(CLASSES))