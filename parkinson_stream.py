
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
model = tf.keras.models.load_model('parkinsons.h5')
def roi(image):
    roi_h = np.array(image)
    roi_j= cv2.cvtColor(roi_h, cv2.COLOR_RGB2BGR)
    roi_g = cv2.resize(roi_j, (224, 224))
    img_array = roi_g/255.0
    roi_1 = np.expand_dims(img_array, axis=0)
    roi_1.shape
    return roi_1
st.title('Wave Image')
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_1 = roi(image)
    predictions = model.predict(img_1)
    if predictions > 0.5:
        st.write(f"Predicted Class: Parkinson")
    elif predictions < 0.5:
        st.write(f"Predicted Class: Healthy")
    else:
        st.write(f"Predicted Class: Invalid")






