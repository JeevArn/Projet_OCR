import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pandas as pd

# Import the OCR functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from systeme.OCR_latin_char.scripts.ocr_script import predict_text as predict_text_latin
from systeme.OCR_tamil_char.src.predict import predict_text as predict_text_tamil
from systeme.OCR_tamil_char.src.model import TamilOCRModel

def load_tamil_mapping(mapping_file: str) -> dict:
    """Load the Tamil character mapping from file."""
    mapping_df = pd.read_csv(mapping_file, sep=';')
    return {idx: unicode_char for idx, unicode_char in zip(mapping_df['class'], mapping_df['unicode'])}

# Streamlit app logic
st.title("OCR System")

# Create tabs for OCR and Documentation
tab1, tab2 = st.tabs(["OCR Interface", "API"])

with tab1:
    st.write("Support for Latin and Handwritten Tamil text recognition")

    # Writing system selection
    writing_system = st.selectbox("Select Writing System", ["Latin", "Handwritten Tamil"])

    # Document type selection
    doc_type = st.selectbox("Select Document Type", ["word", "line", "doc"])

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        image = Image.open(uploaded_file)
        image_path = "temp_image.png"
        image.save(image_path)

        try:
            if writing_system == "Latin":
                # Load the Latin OCR model
                model = load_model('systeme/OCR_latin_char/models/OCR_50000w_10e.h5')

                # Load the label encoder (class names for predictions)
                classes = np.load('systeme/OCR_latin_char/data/classes_1.npy')
                label_encoder = LabelEncoder()
                label_encoder.fit(classes)

                # Perform Latin OCR
                result_text = predict_text_latin(image_path, doc_type, model, label_encoder)

            else:  # Tamil
                # Set device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Load Tamil character mapping
                mapping_file = 'systeme/OCR_tamil_char/tamil-unicode-mapping.txt'
                char_mapping = load_tamil_mapping(mapping_file)
                
                # Initialize Tamil model
                model = TamilOCRModel(num_classes=len(char_mapping))
                
                # Load trained weights
                model_path = 'systeme/OCR_tamil_char/models/best_model.pth'
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device)
                model.eval()

                # Perform Tamil OCR
                result_text = predict_text_tamil(image_path, doc_type, model, char_mapping, device)

            # Display the result
            st.subheader("Extracted Text:")
            st.write(result_text)

            # Display the uploaded image
            st.subheader("Uploaded Image:")
            st.image(image, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        finally:
            # Clean up: delete the temporary file
            if os.path.exists(image_path):
                os.remove(image_path)

    # Add footer with information
    st.markdown("---")
    st.markdown("""
    ### Usage Instructions:
    1. Select the Writing System (Latin or Handwritten Tamil)
    2. Choose the type of document (word, line, or doc)
    3. Upload an image containing text
    4. The system will automatically process and display the extracted text
    """)

with tab2:
    st.header("API Documentation")
    
    st.markdown("""
    ### REST API Usage

    The OCR system provides a REST API built with FastAPI.

    #### Base URL
    ```
    http://127.0.0.1:8000
    ```

    #### POST request

    Performs OCR on an image and returns the extracted text.

    **Request Format:**
    ```json
    {
        "image_path": "path/to/your/image.png",
        "ocr_type": "doc",
        "language": "latin"
    }
    ```

    **Parameters:**
    - `image_path` (string, required): Path to the image file
    - `ocr_type` (string, required): Type of document
        - `"word"`: Single word
        - `"line"`: Single line of text
        - `"doc"`: Full document
    - `language` (string, required): Writing system
        - `"latin"`: Latin script
        - `"tamil"`: Tamil script

    **Response Format:**
    ```json
    {
        "text": "extracted text here",
        "language": "latin"
    }
    ```

    #### Example Usage

    **Using cURL:**

    For Latin text:
    ```bash
    curl -X POST http://127.0.0.1:8000/ocr/ \\
    -H "Content-Type: application/json" \\
    -d '{
        "image_path": "systeme/OCR_latin_char/data/images/doc1.png",
        "ocr_type": "doc",
        "language": "latin"
    }'
    ```

    For Tamil text:
    ```bash
    curl -X POST http://127.0.0.1:8000/ocr/ \\
    -H "Content-Type: application/json" \\
    -d '{
        "image_path": "systeme/OCR_tamil_char/data/images_for_testing/text1.png",
        "ocr_type": "doc",
        "language": "tamil"
    }'
    ```

    **Using Python:**
    ```python
    import requests
    import json

    url = "http://127.0.0.1:8000/ocr/"
    
    data = {
        "image_path": "path/to/your/image.png",
        "ocr_type": "doc",
        "language": "latin"  # or "tamil"
    }

    response = requests.post(url, json=data)
    result = response.json()
    print(result["text"])
    ```

    ### Running the API Server

    To start the API server:
    ```bash
    uvicorn interfaces.api.app:app --reload
    ```
    """)
