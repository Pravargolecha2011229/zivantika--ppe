import os
import streamlit as st
from PIL import Image, ImageFilter, ImageDraw
import numpy as np

def simple_edge_detection(image):
    """Simple edge detection using PIL"""
    # Convert to grayscale
    gray = image.convert('L')
    
    # Apply edge enhancement
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    return edges

def detect_objects_simple(image):
    """Simple object detection simulation"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Draw some simulated detections
    for i in range(3):
        x = random.randint(50, width-100)
        y = random.randint(50, height-100)
        size = random.randint(30, 100)
        
        draw.rectangle([x, y, x+size, y+size], outline='red', width=3)
        draw.text((x+5, y+5), "Object", fill='red')
    
    return image

# Streamlit app
st.title("Simple Image Processing Demo")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image")
    
    with col2:
        edges = simple_edge_detection(image)
        st.image(edges, caption="Edge Detection")
    
    if st.button("Simulate Object Detection"):
        detected = detect_objects_simple(image.copy())
        st.image(detected, caption="Simulated Detections")
