# image_processing.py

from PIL import Image,ImageOps
import cv2 as  cv
import numpy as np
from tensorflow.keras.models import Sequential,load_model
label=['n', '7', 'r', '2', 'b', 'i', 'f', 'h', '5', 'e', 'u', 'm', '8', 'x', '0', 'k', 'q', 'y', 's', 'g', 'a', 'o', 't', 'v', 'z', '3', '1', 'c', '4', 'p', '9', 'l', '6', 'w', 'd', 'j']


def process_image(image_path):
    # Read the image using cv2
    img = cv.imread(image_path)

    # Resize the image to 224x224 using cv2
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)

    # Convert BGR image to RGB (OpenCV uses BGR by default)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Add a batch dimension to the image
    img = np.expand_dims(img, axis=0)

    return img

def fetch_text(img):
    # Replace this with your actual text fetching logic
    x=np.array(img)
    model = load_model('https://drive.google.com/file/d/1mLNfBu6hiqFy7XTQWbV8jOOr3vLhghso/view?usp=sharing')
    p=model.predict(img)
    pi=np.argmax(p, axis=1)
    l=pi[0]
    # For this example, we'll returxn a static text
    return label[l]

# app.py

import streamlit as st
from PIL import Image
import os
import tempfile
def main():
    st.title("Image Processing and Text Fetching App")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_image = tempfile.NamedTemporaryFile(delete=False)
        temp_image.write(uploaded_file.read())

        # Display the uploaded image in the first column
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image using the process_image function
        processed_image = process_image(temp_image.name)

        # Display the processed image in the second column
        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Close and remove the temporary file
        temp_image.close()
        os.remove(temp_image.name)

    # Fetch text (replace this with your actual text fetching logic)
        text = fetch_text(processed_image)

        # Display the fetched text
        st.text("Fetched Text:")
        st.write(text)

if __name__ == "__main__":
    main()
