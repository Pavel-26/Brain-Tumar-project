import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2

# Load model
model = load_model("E:\Brain Tumar\model.h5")

class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Function to increase contrast
def increase_contrast(image, alpha=1.7):
    return cv2.convertScaleAbs(image, alpha=alpha)

# Function to apply mask 
def apply_mask(image, threshold=128):
    # Convert to binary mask using a threshold
    _, binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
    return masked_image

# Function to preprocess 
def preprocess_image(img_path, image_size=(256, 256)):
    # Load the image
    img = np.array(Image.open(uploaded_file))
    if img is None:
        raise ValueError("Image not found or unable to load.")
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, image_size)
    img_normalized = np.array(img_resized) / 255.0
    img_contrasted = increase_contrast(img_resized)
    
    img_masked = apply_mask(img_contrasted)
    
    return img_resized, img_contrasted, img_masked, img_normalized

def prepare_image_for_prediction(image):
    image = np.expand_dims(image, axis=-1)  
    image = np.expand_dims(image, axis=0)  
    return image


# Streamlit UI
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection Web App")
st.markdown("Upload a brain MRI image to check for tumors.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])   

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Preprocessing and analyzing image..."):
        preprocessed_image, contrasted_image, masked_image, normalized_image = preprocess_image(image)
        image_for_prediction = prepare_image_for_prediction(normalized_image)

        # Predict using the model
        prediction = model.predict(image_for_prediction)[0]
        
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]
        predicted_confidence = prediction[predicted_class]

        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {predicted_label}")
        st.info(f"Confidence: {predicted_confidence * 100:.2f}%")