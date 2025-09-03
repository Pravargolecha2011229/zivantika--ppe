import os
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import random
import tempfile

# Set page config
st.set_page_config(page_title="PPE Detection Simulator", layout="wide")
st.title("🦺 PPE Detection Simulator")
st.write("Upload an image to simulate PPE detection (Hardhat, Safety Vest, Person).")

# Simple simulation of object detection
def simulate_ppe_detection(image):
    """Simulate PPE detection by drawing random bounding boxes"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Define PPE classes and colors
    ppe_classes = {
        "Hardhat": "red",
        "Safety Vest": "blue", 
        "Person": "green",
        "No Hardhat": "orange",
        "No Vest": "purple"
    }
    
    # Generate random detections (2-5 objects)
    num_detections = random.randint(2, 5)
    detections = []
    
    for i in range(num_detections):
        # Random bounding box
        box_width = random.randint(50, 200)
        box_height = random.randint(50, 200)
        x = random.randint(0, width - box_width)
        y = random.randint(0, height - box_height)
        
        # Random class
        class_name = random.choice(list(ppe_classes.keys()))
        color = ppe_classes[class_name]
        confidence = round(random.uniform(0.6, 0.95), 2)
        
        # Draw bounding box
        draw.rectangle([x, y, x + box_width, y + box_height], 
                      outline=color, width=3)
        
        # Draw label background
        label = f"{class_name} {confidence}"
        try:
            font = ImageFont.load_default()
            text_width = font.getlength(label)
        except:
            text_width = len(label) * 10
            
        draw.rectangle([x, y-20, x + text_width + 10, y], fill=color)
        
        # Draw label text
        try:
            draw.text((x+5, y-18), label, fill="white", font=font)
        except:
            draw.text((x+5, y-18), label, fill="white")
        
        detections.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": [x, y, x + box_width, y + box_height]
        })
    
    return image, detections

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and display original image
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Original Image", use_container_width=True)
    
    # Process button
    if st.button("🔍 Run PPE Detection Simulation"):
        with st.spinner("Simulating PPE detection..."):
            # Create a copy for processing
            processed_image = original_image.copy()
            
            # Simulate detection
            result_image, detections = simulate_ppe_detection(processed_image)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result_image, caption="Detection Results", use_container_width=True)
            
            with col2:
                st.subheader("Detection Summary")
                st.write(f"**Total detections:** {len(detections)}")
                
                # Count by class
                class_counts = {}
                for detection in detections:
                    class_name = detection["class"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                for class_name, count in class_counts.items():
                    st.write(f"- {class_name}: {count}")
                
                # Show details in expander
                with st.expander("View Detailed Results"):
                    for i, detection in enumerate(detections, 1):
                        st.write(f"**Detection {i}:**")
                        st.write(f"- Class: {detection['class']}")
                        st.write(f"- Confidence: {detection['confidence']:.2f}")
                        st.write(f"- Bounding Box: {detection['bbox']}")
                        st.write("---")

# Add some educational content
st.markdown("---")
st.subheader("About PPE Detection")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **PPE (Personal Protective Equipment) Detection**
    
    This simulator demonstrates how computer vision can detect:
    - Safety helmets (Hardhats)
    - Safety vests
    - Personnel on site
    - Compliance violations
    
    In a real implementation, this would use YOLO or similar AI models.
    """)

with col2:
    st.warning("""
    **For Real Detection:**
    
    1. Run locally with proper dependencies:
    ```bash
    pip install ultralytics opencv-python
    ```
    
    2. Or use cloud services like:
    - Google Cloud Vision AI
    - AWS Rekognition
    - Azure Computer Vision
    """)

# Footer
st.markdown("---")
st.caption("Note: This is a simulation. For actual PPE detection, run the application locally with proper dependencies.")
