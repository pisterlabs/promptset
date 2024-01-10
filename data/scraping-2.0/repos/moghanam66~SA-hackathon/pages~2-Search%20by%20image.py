import os
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from PIL import Image
import openai
from utils import *

def main():

    # Congfiger the page attributes
    st.set_page_config(
        page_title='Search by image',
        page_icon=":star:",layout="wide")
    
    # Set the background and the logo
    set_background('wallpaper.jpeg')
    add_logo("logo4.png")
    
    # Set the title
    st.title("Image Similarity Search")

    # Set a css style for markdown to increase its font size
    st.markdown("""
    <style>
    .big-font {
        font-size:25px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('''<p class="big-font">Have you visited a place and asked yourself if there is someplace like this in Saudi Arabia? Or are you searching for a specific vibe, and where can you find it in Saudi Arabia?
    Upload a photo of your desired location, and we'll help you find the most similar place to it and where to find it.</p>''', unsafe_allow_html=True)

    # Read the scrapped data of the attraction sites
    df = pd.read_csv('attractionsEng.csv')

    # Get the image from the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg",'png','jpeg'])

    # Load the MobileNetV2 model from TensorFlow Hub
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    model = tf.keras.Sequential([hub.KerasLayer(model_url, input_shape=(224, 224, 3))])

    #Create a vector reparesnting the images in the folder cvs_images to compare it with the input image
    annoy_index,image_files,image_folder=loadImages(model)

    if uploaded_file is not None:

        # load the input image and display it
        user_img = Image.open(uploaded_file)
        _,col2,_=st.columns([0.2,0.4,0.2])
        with col2:
            st.image(user_img, caption="Uploaded Image", use_column_width=True)

        # preproces the image and predict the vector representing the image
        user_img = load_and_preprocess_image(user_img)
        user_features = model.predict(np.expand_dims(user_img, axis=0))

        # Find the most similar image in the dataset and load it
        n_nearest_neighbors = 1
        nearest_indices = annoy_index.get_nns_by_vector(user_features.squeeze(), n_nearest_neighbors)
        nearest_image_file = image_files[nearest_indices[0]]
        similar_image = Image.open(os.path.join(image_folder, nearest_image_file))

        # Represent a similar image and the location of the place presented by the image
        col1,col2=st.columns([0.5,0.5])
        with col1:
            st.image(similar_image, caption="Similar Image", use_column_width=True)
        with col2:
            # get the row of the dataset representing the attraction site
            index_without_extension = int(nearest_image_file.split('.')[0])
            row = df.loc[index_without_extension]

            # Define the coordinates (latitude, longitude) of the place
            coordinates = {'latitude': [row['Latitude']], 'longitude': [row['Longitude']]}

            # Create a DataFrame with the coordinates
            df_coordinates = pd.DataFrame(coordinates)

            # Display the map centred on the coordinates
            st.map(df_coordinates)
        
        #Display all the info about the place from the scrapped data
        sentence = f"The place : {row['attractionSite']}, it's located in  {row['city']}, {row['description']}."
        st.title(sentence)

        # Display MORE information through chat GPT
        st.title(openAiDescription(row['attractionSite']))

# Function to load and preprocess images
def load_and_preprocess_image(image):
    img = image.resize((224, 224))
    img = img.convert("RGB")  # Convert image to RGB if it has an alpha channel
    img = np.array(img) / 255.0  # Normalize image pixels to [0, 1]
    return img

# Casche the features of each image to increase the performance
@st.cache_resource
def loadImages(_model):
    # Load dataset images and extract features
    image_folder = 'csv_images'
    image_files = os.listdir(image_folder)

    image_features = []
    for image_file in image_files:
        img = Image.open(os.path.join(image_folder, image_file))
        img = load_and_preprocess_image(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        features = _model.predict(img) # predict the features
        image_features.append(features.squeeze())

    # Build Annoy index
    annoy_index = AnnoyIndex(1280, "euclidean")
    for i, feature in enumerate(image_features):
        annoy_index.add_item(i, feature)
    annoy_index.build(10)

    return annoy_index,image_files,image_folder

# Cache the answer to optimize the performance and reduce any wasted cost. 
@st.cache_data
def openAiDescription(place):
    openai.api_key = OPENAI_API_KEY 
    # Set the model and prompt
    model_engine = "text-davinci-003"
    prompt =  f"Give a description about {place} in less than 70 words about in Saudi Arabia"

    # Set the maximum number of tokens to generate in the response
    max_tokens = 1024
    # Generate a response
    completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    return completion.choices[0].text


if __name__ == "__main__":
    main()  
