import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Streamlit page configuration
st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_LOCAL_PATH = ROOT / "models/YOLOv11x_RDD_Trained.pt"

# Logger setup
logger = logging.getLogger(__name__)

# Load the model from local path (no downloading)
cache_key = "yolov11xrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

# Class labels
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

# Title and instructions
st.title("Road Damage Detection - Image")
st.write("Upload one or more images to detect road damage. The model will identify types of cracks and potholes.")

# Upload widget: allow multiple files
image_files = st.file_uploader("Upload Images", type=['png', 'jpg'], accept_multiple_files=True)

# Confidence threshold slider
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
st.write("Adjust the threshold to control detection sensitivity.")

# Loop over all uploaded images
if image_files:
    for idx, image_file in enumerate(image_files):
        st.markdown(f"---\n### Image {idx + 1}: {image_file.name}")
        image = Image.open(image_file)
        _image = np.array(image)
        h_ori, w_ori = _image.shape[:2]

        # Resize and predict
        image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
        results = net.predict(image_resized, conf=score_threshold)

        # Process detections
        for result in results:
            boxes = result.boxes.cpu().numpy()
            detections = [
                Detection(
                    class_id=int(_box.cls),
                    label=CLASSES[int(_box.cls)],
                    score=float(_box.conf),
                    box=_box.xyxy[0].astype(int),
                )
                for _box in boxes
            ]

        # Plot predictions and resize back to original
        annotated_frame = results[0].plot()
        _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

        # Display original and predicted images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Image**")
            st.image(_image)

        with col2:
            st.write("**Prediction**")
            st.image(_image_pred)

            # Download button
            buffer = BytesIO()
            Image.fromarray(_image_pred).save(buffer, format="PNG")
            st.download_button(
                label="Download Prediction Image",
                data=buffer.getvalue(),
                file_name=f"{Path(image_file.name).stem}_prediction.png",
                mime="image/png"
            )
