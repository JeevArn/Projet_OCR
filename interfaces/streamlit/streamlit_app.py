import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Import the OCR functions from ocr_script.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from systeme.scripts.ocr_script import predict_text, predict_word

# Streamlit app logic
st.title("OCR Model")

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Document type selection
doc_type = st.selectbox("Select Document Type", ["word", "line", "doc"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    image = Image.open(uploaded_file)
    image_path = "temp_image.png"
    image.save(image_path)

    # Load the OCR model
    model = load_model('systeme/models/OCR_50000w_10e.h5')

    # Load the label encoder (class names for predictions)
    classes = np.load('systeme/data/classes_1.npy')
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    # Perform OCR on the uploaded image
    result_text = predict_text(image_path, doc_type, model, label_encoder)

    # Display the result
    st.subheader("Extracted Text:")
    st.write(result_text)

    # Optionally, delete the temporary file after processing
    os.remove(image_path)
