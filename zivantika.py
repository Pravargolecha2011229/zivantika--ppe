import os
import sys
import streamlit as st
import subprocess
import importlib.util

# Set environment variables before any other imports
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["PYTHONPATH"] = "/home/adminuser/venv/lib/python3.11/site-packages"

def install_system_packages():
    """Install system packages required for OpenCV"""
    packages = [
        "libglib2.0-0",
        "libsm6", 
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgl1-mesa-glx",
        "ffmpeg",
        "libsm6",
        "libxext6"
    ]
    
    try:
        for package in packages:
            subprocess.run(["apt-get", "install", "-y", package], 
                         capture_output=True, check=True)
    except:
        pass  # Ignore errors in system package installation

def check_and_install_packages():
    """Check and install required Python packages"""
    required_packages = {
        'ultralytics': 'ultralytics==8.0.196',
        'cv2': 'opencv-python-headless==4.8.1.78',
        'PIL': 'Pillow',
        'torch': 'torch',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, install_name in required_packages.items():
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(install_name)
    
    if missing_packages:
        st.error(f"Missing packages: {', '.join(missing_packages)}")
        st.info("Installing required packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                st.success(f"Installed {package}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install {package}: {e}")
        
        st.info("Please refresh the page after installation completes.")
        st.stop()

# Install system packages
install_system_packages()

# Check and install Python packages
check_and_install_packages()

# Now try to import required libraries
try:
    import cv2
    import numpy as np
    from PIL import Image
    import tempfile
    from ultralytics import YOLO
    
    st.success("✅ All packages imported successfully!")
    
except ImportError as e:
    st.error(f"❌ Import error: {str(e)}")
    st.info("""
    **To fix this error:**
    
    1. Create a `packages.txt` file in your repository root with:
    ```
    libglib2.0-0
    libsm6
    libxext6
    libxrender-dev
    libgomp1
    libgl1-mesa-glx
    ffmpeg
    ```
    
    2. Create a `requirements.txt` file with:
    ```
    streamlit
    ultralytics==8.0.196
    opencv-python-headless==4.8.1.78
    Pillow>=9.0.0
    torch>=1.11.0
    torchvision>=0.12.0
    numpy>=1.21.0
    ```
    
    3. Redeploy your Streamlit app
    """)
    st.stop()

@st.cache_resource
def load_model():
    """Load the YOLO model with caching to avoid reloading on every interaction"""
    MODEL_PATH = "runs/detect/train/weights/best.pt"
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.info("""
        **Model file missing!**
        
        Please ensure your trained model is placed at: `runs/detect/train/weights/best.pt`
        
        You can either:
        1. Upload your trained model to this path in your repository
        2. Or modify the MODEL_PATH variable to point to your model location
        """)
        st.stop()
    
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Make sure your model file is a valid YOLO model (.pt file)")
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
            
            with st.spinner("Processing image..."):
                results = model.predict(image_array, verbose=False)
            
            # Show output
            if results and len(results) > 0:
                result_img = results[0].plot()  # numpy array (BGR)
                # Convert BGR to RGB for proper display in Streamlit
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, caption="Detection Result", use_container_width=True)
                
                # Show detection statistics
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    st.info(f"🎯 Found {len(detections)} detections")
                else:
                    st.warning("⚠️ No objects detected in the image")
            else:
                st.warning("⚠️ No detections found in the image.")
                
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try with a different image or check if the model is compatible.")

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
            with st.spinner("Processing video... This may take a while for large files."):
                results = model.predict(
                    tmp_file_path, 
                    save=True, 
                    project="runs/streamlit_results", 
                    name="ppe_video",
                    verbose=False
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
                pass # Ignore cleanup errors
                
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
            st.info("Please try with a smaller video file or different format.")
            
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
    **Common Issues:**
    - **Model not found**: Ensure your trained model is at `runs/detect/train/weights/best.pt`
    - **Import errors**: Make sure `packages.txt` and `requirements.txt` are properly configured
    - **Video processing fails**: Try with a smaller video file first
    - **No detections**: The model may not detect PPE if confidence is too low
    
    **Required Files for Deployment:**
    - `packages.txt` (for system packages)
    - `requirements.txt` (for Python packages)
    - Your trained model at the correct path
    """)

with st.expander("📋 Setup Files"):
    st.markdown("**packages.txt:**")
    st.code("""libglib2.0-0
libsm6
libxext6
libxrender-dev
libgomp1
libgl1-mesa-glx
ffmpeg""")
    
    st.markdown("**requirements.txt:**")
    st.code("""streamlit
ultralytics==8.0.196
opencv-python-headless==4.8.1.78
Pillow>=9.0.0
torch>=1.11.0
torchvision>=0.12.0
numpy>=1.21.0""")
