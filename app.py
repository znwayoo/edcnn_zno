import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('/Users/zno/Git/edcnn_zno/model/initial_model.keras')

# Define the class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((48, 48)).convert('L')  # Resize and convert to grayscale
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255  # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title('Emotion Detection')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load 
    img = Image.open(uploaded_file)
    
    # Preprocess the image
    processed_image = preprocess_image(img)
    
    # Make prediction
    prediction = model.predict(processed_image)
    emotion = class_labels[np.argmax(prediction)]
    
    st.write(f'Predicted Emotion: {emotion}')
    
    #Display Image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
