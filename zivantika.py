import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# 🔧 Fix Streamlit + Torch watcher bug
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"

# Load your trained model (update path if needed)
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# Streamlit UI
st.set_page_config(page_title="PPE Detection", layout="wide")
st.title("🦺 PPE Detection App")
st.write("Upload an image or video to detect PPE (Hardhat, Safety Vest, Person).")

# File uploader
uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    file_type = uploaded_file.type

    # Handle Images
    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run YOLOv8 inference
        st.write("🔎 Running detection...")
        results = model.predict(image)
        
        # Show output
        result_img = results[0].plot()  # numpy array (BGR)
        st.image(result_img, caption="Detection Result", use_container_width=True)

    # Handle Images
    if file_type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run YOLOv8 inference
        st.write("🔎 Running detection...")
        results = model.predict(image)

        # Show output directly
        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_container_width=True)

    # Handle Videos
    elif file_type.startswith("video"):
        st.video(uploaded_file)
        st.write("⚡ Running detection on video...")

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        results = model.predict("temp_video.mp4", save=True, project="runs/streamlit_results", name="ppe_video")

        st.success("✅ Video processed. Check saved results in `runs/streamlit_results/ppe_video/`")