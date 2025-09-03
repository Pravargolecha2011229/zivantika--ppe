import os
import streamlit as st
import tempfile
from PIL import Image
import subprocess
import sys

# 🔧 Set environment variables to prevent GUI dependencies
os.environ["STREAMLIT_WATCHER_IGNORE"] = "tornado,torch"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Prevent Qt GUI dependencies
os.environ["PYTHONPATH"] = "/home/adminuser/venv/lib/python3.11/site-packages"

# Try to install headless OpenCV first
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "opencv-python-headless==4.5.5.64", "--force-reinstall"])
    st.success("Installed opencv-python-headless")
except Exception as e:
    st.warning(f"Could not install headless OpenCV: {e}")

# Now try to import ultralytics
try:
    from ultralytics import YOLO
    ULTRAlytics_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import ultralytics: {e}")
    ULTRAlytics_AVAILABLE = False
except OSError as e:
    if "libGL.so.1" in str(e):
        # Try to work around the OpenCV issue
        st.warning("OpenCV GUI dependencies missing. Trying alternative approach...")
        try:
            # Force use of headless backend
            import cv2
            cv2.setUseOptimized(True)
            # Now retry ultralytics import
            from ultralytics import YOLO
            ULTRAlytics_AVAILABLE = True
            st.success("Successfully imported ultralytics with workaround!")
        except Exception as inner_e:
            st.error(f"Still failed to import: {inner_e}")
            ULTRAlytics_AVAILABLE = False
    else:
        st.error(f"OS error: {e}")
        ULTRAlytics_AVAILABLE = False

# Streamlit UI
st.set_page_config(page_title="PPE Detection", layout="wide")
st.title("🦺 PPE Detection App")
st.write("Upload an image or video to detect PPE (Hardhat, Safety Vest, Person).")

if not ULTRAlytics_AVAILABLE:
    st.error("""
    **Critical: Unable to load computer vision dependencies.**
    
    This is likely due to missing system libraries in the Streamlit Cloud environment.
    Please try these alternatives:
    
    1. **Use a different model format**: Convert your YOLO model to ONNX or TensorFlow Lite
    2. **Use a different detection library**: Consider using TensorFlow or PyTorch directly
    3. **Local deployment**: Run this app on your local machine instead of Streamlit Cloud
    
    For immediate testing, you can use a simplified version without ultralytics:
    """)
    
    # Simple file upload demo without ultralytics
    uploaded_file = st.file_uploader("Upload Image (Demo Mode)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image (Demo Mode - No Detection)", use_container_width=True)
        st.info("In demo mode - detection would normally happen here with ultralytics")
    st.stop()

# Load your trained model
MODEL_PATH = "runs/detect/train/weights/best.pt"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.warning(f"Model file not found at: {MODEL_PATH}")
    st.info("Using default YOLOv8 model for demonstration...")
    try:
        model = YOLO('yolov8n.pt')  # Use default nano model
        st.success("Loaded default YOLOv8n model for demonstration")
    except Exception as e:
        st.error(f"Could not load default model: {e}")
        st.stop()
else:
    try:
        model = YOLO(MODEL_PATH)
        st.success("Custom model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load custom model: {e}")
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
            results = model.predict(image, conf=0.5)
            
            # Show output
            result_img = results[0].plot()  # numpy array (BGR)
            st.image(result_img, caption="Detection Result", use_container_width=True)
            
            # Display detection summary
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                detected_count = len(results[0].boxes)
                st.write(f"✅ Detected {detected_count} objects")
                
                # Show class names if available
                if hasattr(results[0], 'names'):
                    class_counts = {}
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        class_name = results[0].names[cls]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    for class_name, count in class_counts.items():
                        st.write(f"- {class_name}: {count}")
                        
        except Exception as e:
            st.error(f"Error during detection: {e}")

    # Handle Videos
    elif file_type.startswith("video"):
        st.video(uploaded_file)
        st.info("Video processing is disabled in this environment due to library limitations.")
        st.write("For video processing, please run this application locally on your machine.")
