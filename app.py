import streamlit as st
import cv2
import numpy as np
from src.pipeline.predict_pipeline import predict_image

def main():

    st.title("Image Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        st.image(cv2.cvtColor(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

        predicted_class = predict_image(uploaded_file)
        st.success(f"Predicted Class: {predicted_class}")

if __name__=="__main__":
    main()