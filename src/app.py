import sys
import os

# Add src/ to the Python path
sys.path.append(os.path.abspath("src"))

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.inference import generate_caption, preprocess_image

# Paths
MODEL_SAVE_DIR = "models/"
PREPROCESSED_DIR = "data/preprocessed/"
model_path = os.path.join(MODEL_SAVE_DIR, "image_captioning_model_custom.h5")  # Change to "pretrained" if needed

# Load tokenizer
with open(os.path.join(PREPROCESSED_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# Load trained model
model = load_model(model_path, custom_objects={"build_captioning_model": generate_caption})

# Streamlit UI
st.title("🖼️ AI Image Caption Generator")

st.markdown("""
### Upload an image to generate a caption:
""")

# Upload file UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_array = preprocess_image(uploaded_file)

    # Generate a caption
    with st.spinner("Generating caption..."):
        caption = generate_caption(model, image_array, tokenizer)

    st.subheader("Generated Caption:")
    st.write(f"**{caption}**")

st.markdown("""
### Instructions:
1. Upload an image using the file uploader.
2. Wait for the model to process and generate a caption.
3. The generated caption will be displayed below the image.
""")
