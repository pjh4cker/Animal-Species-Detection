import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Title and Header
st.title("Animal Species Detection App")
st.header("Upload an image and the app will predict the animal species.")
st.text("Created by: Jay Parmar, Viraj Parmar, Hemang Khatri, and Dhyan Patel")

# Load the model
model_path = 'D:\Minor Project\ASD\model\Animal_Prediction.h5'
model = tf.keras.models.load_model(model_path)

# Define function to get label
def get_key(index):
    labels = ['Chicken', 'Deer', 'Duck', 'Eagle', 'Fish', 'Frog', 'Horse', 'Lizard', 'Monkey', 'Penguin', 'Shark', 'Snake', 'Sparrow', 'Spider', 'Tiger', 'Tortoise']
    return labels[index]

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# If image uploaded
if uploaded_file is not None:
    # Display the uploaded image
    test_image = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(test_image, caption='Uploaded Image', use_column_width=True)
    
    # Load the image
    test_image_array = image.img_to_array(test_image)
    test_image_array = np.expand_dims(test_image_array, axis=0)

    # Make prediction
    try:
        result = model.predict(test_image_array)
        max_index = np.argmax(result)
        predicted_label = get_key(max_index)
        accuracy = result[0][max_index] * 100  # Confidence score for the predicted class

        # Display the predicted class label and accuracy
        st.write(f"Predicted animal species: {predicted_label} (Accuracy: {accuracy:.2f}%)")
    except Exception as e:
        st.error(f"Error predicting: {e}")
