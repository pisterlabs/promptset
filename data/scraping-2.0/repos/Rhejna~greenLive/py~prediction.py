import pickle
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import lightgbm
import sklearn
from PIL import Image
import pandas as pd

import openai
# Configure the API_KEY
openai.api_key = api_key = API_KEY = "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"
model = "gpt-3.5-turbo"

import os
# # !pip install opencv-python
# import cv2  # You may need to install OpenCV (cv2) if not already installed


################ CROP RECOMMANDATION ###########################
def recommndant_crop(config):
    # loading the model from the saved file
    # pkl_filename = "../models/model_recommandation.pkl"
    ROOT_DIR = os.path.abspath(os.curdir)
    print(ROOT_DIR)
    pkl_filename = os.path.join(ROOT_DIR, 'models/model_recommandation.pkl')
    # pkl_filename = os.path.join(ROOT_DIR, '../models/model_recommandation.pkl')
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    result = [[value for value in config.values()]]
    print(result)

    # if type(config) == dict:
    #     df = pd.DataFrame(config)
    # else:
    #     df = config

    y_pred = model.predict(result)

    return y_pred


################ DISEASE PREDICTION ###########################
def predict_disease(config):
    ##loading the model from the saved file
    ROOT_DIR = os.path.abspath(os.curdir)
    print(ROOT_DIR)
    pkl_filename = os.path.join(ROOT_DIR, 'models/potatoes.h5')
    # pkl_filename = os.path.join(ROOT_DIR, '../models/potatoes.h5')

    model = models.load_model(pkl_filename)

    IMAGE_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3
    EPOCHS = 50

    print(config)

    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    try:
        # Open the image
        img = Image.open(config)

        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert to a NumPy array
        image_array = np.array(img)

        # image_array = tf.keras.preprocessing.image.img_to_array(dataset).astype('uint8')
    except Exception as e:
        return f"Error loading image: {str(e)}"

    # Make predictions without verbose output
    predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)

    # Extract the predicted class index and confidence (probability)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100

    print(predicted_class_index)

    if predicted_class_index == 0 or predicted_class_index == 1:
        y_pred = f"The plant is sick. Predicted class label: {class_names[predicted_class_index]}, (Confidence: {confidence:.2f}%)"
    elif predicted_class_index == 2:
        y_pred = f"The plant is healthy. Predicted class label: {class_names[predicted_class_index]}, (Confidence: {confidence:.2f}%)"

    return y_pred


################ WEED PREDICTION ###########################
def predict_weed(config):
    ROOT_DIR = os.path.abspath(os.curdir)
    pkl_filename = os.path.join(ROOT_DIR, 'models/model_weed.pkl')
    # pkl_filename = os.path.join(ROOT_DIR, '../models/model_weed.pkl')
    with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)

    # Image size that we are going to use
    IMG_SIZE = 128

    # Load an individual plant image
    image_path = config  # Replace with the path to your image
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), interpolation='bilinear')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Preprocess the image
    img_array = img_array / 255.0  # Normalize

    # Make predictions using the model
    predictions = model.predict(img_array)
    confidence_percentage = predictions[0][0] * 100  # Convert to percentage

    # Get the predicted class name
    predicted_class = "Potato" if confidence_percentage < 50 else "Weed"  # Assuming 50% threshold

    # Display results
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Percentage: {confidence_percentage:.2f}%")

    return f"Predicted: {predicted_class}, Confidence: {confidence_percentage:.2f}%"


################ FERTILIZER RECOMMANDATION ###########################
def predict_fertilizer_amount(data: list[str]):

    ROOT_DIR = os.path.abspath(os.curdir)
    pkl_filename = os.path.join(ROOT_DIR, 'models/fertilizer.pkl')
    #  loading the model from the saved file
    with open(pkl_filename, 'rb') as f_in:
        fertilizer_model = pickle.load(f_in)

    # if type(config) == dict:
    #     df = pd.DataFrame(config)
    # else:
    #     df = config

    #  make the prediction
    return fertilizer_model.predict(pd.DataFrame(data, index=[0]))


################ FARMER PERSONAL ASSISTANT ###########################è
def answer(config):
    # Personality
    identity = "Gérome"
    creators = "AI developpers and experienced farmers from GREENLIVE"
    mission = f"an experienced AI farmer developped by {creators} and your role is to help farmers to understand the data in their farm"
    # Context
    context = {
        "role": "system",
        "content": f"Your name is {identity}. You where created by {creators}. You are {mission}."
    }

    # Provide the context initially
    messages = [context]

    messages.append({
        "role": "user",
        "content": config
    })

    # Prompt chatGPT
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.9,
        max_tokens=150,
    )
    # Extract the reply
    reply = chat.choices[0].message.content
    # print the message
    reply = reply.replace('. ', '.\n')
    print(f"Gerome : {reply}\n")
    return reply