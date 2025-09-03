import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

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

    # Handle Videos
    elif file_type.startswith("video"):
        st.video(uploaded_file)
        st.write("⚡ Running detection on video...")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name

        try:
            # Run prediction
            results = model.predict(tmp_file_path, save=True, project="runs/streamlit_results", name="ppe_video")
            
            # Find the output video
            output_dir = "runs/streamlit_results/ppe_video"
            if os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                video_files = [f for f in output_files if f.endswith('.mp4')]
                if video_files:
                    output_video_path = os.path.join(output_dir, video_files[0])
                    st.success("✅ Video processed successfully!")
                    st.video(output_video_path)
                else:
                    st.warning("⚠️ No output video found in results directory")
            else:
                st.error("❌ Output directory not found")
                
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)