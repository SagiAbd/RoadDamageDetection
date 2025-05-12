import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection App",
    page_icon="🛣️",
)

# Banner
st.divider()
st.title("Road Damage Detection App")

# Description
st.markdown(
    """
    Welcome to the Road Damage Detection App — an AI-powered tool that detects and classifies road surface damage using images, videos, or live webcam input.

    This app uses a YOLOv8 deep learning model trained on real-world road imagery from Japan and India to accurately detect four common types of road damage:

    - **Longitudinal Crack**
    - **Transverse Crack**
    - **Alligator Crack**
    - **Potholes**

    The model is optimized for fast and efficient detection, supporting various input modes to suit different needs — whether you're analyzing a single image or inspecting roads in real time.

    ---
    ### How to Use
    - Use the **sidebar** to select your input method:
      - Image Upload
    - Adjust the detection confidence threshold as needed.
    - View annotated results and download them if required.

    ---
    ### Resources
    - Model: YOLOv8 Small — [Ultralytics](https://github.com/ultralytics/ultralytics)
    - Built with [Streamlit](https://streamlit.io/)
    """
)
