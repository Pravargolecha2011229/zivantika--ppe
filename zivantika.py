import os
import streamlit as st
import tempfile
from PIL import Image

# 🔧 Fix Streamlit + Torch watcher bug and other environment issues
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Avoid OpenEXR issues

# Try to import ultralytics with error handling
try:
    from ultralytics import YOLO
    ULTRAlytics_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import ultralytics: {e}")
    ULTRAlytics_AVAILABLE = False
except OSError as e:
    st.error(f"OS error when importing ultralytics: {e}")
    ULTRAlytics_AVAILABLE = False

# Streamlit UI
st.set_page_config(page_title="PPE Detection", layout="wide")
st.title("🦺 PPE Detection App")
st.write("Upload an image or video to detect PPE (Hardhat, Safety Vest, Person).")

# Show warning if ultralytics is not available
if not ULTRAlytics_AVAILABLE:
    st.warning("""
    **Dependencies not fully loaded.** 
    This may be due to missing system libraries. 
    Trying to install required packages...
    """)
    
    # Try to install required packages
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                              "opencv-python-headless==4.5.5.64", "ultralytics==8.3.191"])
        st.success("Packages installed successfully! Please refresh the page.")
    except Exception as e:
        st.error(f"Failed to install packages: {e}")
    st.stop()

# Load your trained model (update path if needed)
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.info("Please make sure your model is available at the specified path.")
    st.stop()

try:
    model = YOLO(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

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
        try:
            results = model.predict(image)
            
            # Show output
            result_img = results[0].plot()  # numpy array (BGR)
            st.image(result_img, caption="Detection Result", use_container_width=True)
            
            # Display detection summary
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                st.write(f"Detected {len(results[0].boxes)} objects")
        except Exception as e:
            st.error(f"Error during detection: {e}")

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
                
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
