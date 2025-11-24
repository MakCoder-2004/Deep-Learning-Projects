# Streamlit Application

import os
import io
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Image Classifier (CIFAR10)", layout="wide")

st.title("Image Classification")
st.title("Upload an image and the model will predict what is it.")
st.write("The Model Can Predict Planes, Cars, Birds, Cats, Deers, Dogs, Frogs, Horses, Ships and Trucks")

# Model
MODEL_FILE = '../model/best_model.h5'
def _load_model(path):
    return tf.keras.models.load_model(path, compile=False)

try:
    load_model = st.cache_resource(_load_model)
except AttributeError:
    load_model = st.cache(allow_output_mutation=True)(_load_model)

model = load_model(MODEL_FILE)

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# file uploader
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

col1, col2 = st.columns([1, 1])

if uploaded is not None:
    try:
        bytes_data = uploaded.read()
        img = Image.open(io.BytesIO(bytes_data)).convert('RGB')
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a PNG or JPG image.")
        st.stop()

    # show original image
    with col1:
        st.subheader("Original image")
        st.image(img, use_container_width=True)

    # Preprocess: resize to 32x32 and normalize by 255.0
    target_size = (32, 32)
    img_resized = img.resize(target_size, Image.BILINEAR)
    img_array = np.asarray(img_resized).astype('float32') / 255.0

    # Model expects shape (1,32,32,3)
    input_tensor = np.expand_dims(img_array, axis=0)

    # show preprocessed image (enlarged for visibility)
    with col2:
        st.subheader("Preprocessed (32x32) image")
        preview = img_resized.resize((160, 160), Image.NEAREST)
        st.image(preview, use_container_width=False)

    # Predict
    with st.spinner("Running prediction..."):
        preds = model.predict(input_tensor)

    probs = preds[0]
    top_idx = np.argmax(probs)
    predicted_label = class_names[top_idx]

    st.markdown("## Prediction :")
    st.success(f"{predicted_label}")

    # Show probabilities as dataframe
    df = pd.DataFrame({"class": class_names, "probability": probs})
    df = df.sort_values(by='probability', ascending=False).reset_index(drop=True)

    st.subheader("Class probabilities")
    st.bar_chart(df.set_index('class'))

else:
    st.info("Upload an image to get a prediction.")
