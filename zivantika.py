import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np

# 🔧 Fix Streamlit + Torch watcher bug
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"

@st.cache_resource
def load_model():
    """Load the YOLO model with caching to avoid reloading on every interaction"""
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.info("Please ensure your trained model is placed at the correct path.")
        st.stop()
    
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load your trained model
model = load_model()

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
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Run YOLOv8 inference
            st.write("🔎 Running detection...")
            
            # Convert PIL image to format that YOLO can process
            image_array = np.array(image)
            results = model.predict(image_array)
            
            # Show output
            if results and len(results) > 0:
                result_img = results[0].plot()  # numpy array (BGR)
                # Convert BGR to RGB for proper display in Streamlit
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, caption="Detection Result", use_container_width=True)
            else:
                st.warning("⚠️ No detections found in the image.")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

    # Handle Videos
    elif file_type.startswith("video"):
        try:
            st.video(uploaded_file)
            st.write("⚡ Running detection on video...")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Create output directory if it doesn't exist
            output_base = "runs/streamlit_results"
            os.makedirs(output_base, exist_ok=True)

            # Run prediction with progress indicator
            with st.spinner("Processing video..."):
                results = model.predict(
                    tmp_file_path, 
                    save=True, 
                    project="runs/streamlit_results", 
                    name="ppe_video"
                )
                
                # Find the output video
                output_dir = "runs/streamlit_results/ppe_video"
                if os.path.exists(output_dir):
                    output_files = os.listdir(output_dir)
                    video_files = [f for f in output_files if f.endswith('.mp4')]
                    if video_files:
                        output_video_path = os.path.join(output_dir, video_files[0])
                        st.success("✅ Video processed successfully!")
                        
                        # Display the processed video
                        with open(output_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                    else:
                        st.warning("⚠️ No output video found in results directory")
                        st.info("The detection may have completed but the output video wasn't saved properly.")
                else:
                    st.error("❌ Output directory not found")
                    
            # Clean up temporary file
            try:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except Exception as cleanup_error:
                # Don't show error to user for cleanup issues
                pass
                
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            
            # Clean up temporary file in case of error
            try:
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except:
                pass

# Add some helpful information
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Upload an image or video** using the file uploader above
    2. **Supported formats**: 
       - Images: JPG, JPEG, PNG
       - Videos: MP4
    3. **Detection classes**: The model detects Hardhat, Safety Vest, and Person
    4. **Processing time** may vary based on file size and complexity
    """)

with st.expander("🛠️ Troubleshooting"):
    st.markdown("""
    - **Model not found**: Ensure your trained model is at `runs/detect/train/weights/best.pt`
    - **Video processing fails**: Try with a smaller video file first
    - **No detections**: The model may not detect PPE if confidence is too low
    """)
