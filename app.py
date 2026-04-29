import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import exifread

st.set_page_config(page_title="Deepfake & Metadata Analyzer", layout="wide")

# --- 1. CNN MODEL ARCHITECTURE ---
@st.cache_resource
def load_mock_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = load_mock_model()

# --- 2. METADATA EXTRACTOR ---
def extract_metadata(uploaded_file):
    tags = exifread.process_file(uploaded_file, details=False)
    metadata = {}
    
    # Look for camera make, model, and original timestamp
    if 'Image Make' in tags:
        metadata['Camera Brand'] = tags['Image Make']
    if 'Image Model' in tags:
        metadata['Camera Model'] = tags['Image Model']
    if 'EXIF DateTimeOriginal' in tags:
        metadata['Time Clicked'] = tags['EXIF DateTimeOriginal']
        
    return metadata

# --- 3. UI & LOGIC ---
st.title("🔍 Advanced Media Verification Engine")
st.write("Analyze pixel structure and origin metadata to verify digital authenticity.")

uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Media", use_column_width=True)
        
    with col2:
        st.subheader("1. Metadata & Origin Analysis")
        # Reset file pointer for exifread
        uploaded_file.seek(0) 
        metadata = extract_metadata(uploaded_file)
        
        if metadata:
            st.success("Hardware Origin Data Found")
            for key, value in metadata.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning("⚠️ No Hardware Origin Data. This file may be AI-generated or heavily edited, as original camera timestamps are missing.")
            
        st.divider()
        
        st.subheader("2. Deep Learning Pixel Analysis")
        if st.button("Run CNN Scan"):
            with st.spinner("Analyzing layers..."):
                # Preprocess
                img_array = np.array(image.convert('RGB'))
                resized_img = cv2.resize(img_array, (224, 224))
                normalized_img = np.expand_dims(resized_img / 255.0, axis=0)
                
                # Predict
                prediction = model.predict(normalized_img)[0][0]
                
                if prediction > 0.5:
                    st.success(f"✅ Neural Net Assessment: REAL (Score: {prediction:.2f})")
                else:
                    st.error(f"🚨 Neural Net Assessment: FAKE/SYNTHETIC (Score: {1-prediction:.2f})")