# Import libraries
import openai
from dotenv import load_dotenv
import os
import tensorflow as tf
import numpy as np
import cv2

# Define class labels
classLabels = ["human", "cheetah", "parrot"]

# Load trained model
model = tf.keras.models.load_model("classifiers/bioModel")

# Define API key
load_dotenv("Vars.env")
key = os.getenv("APIKEY")
openai.api_key = key

# Get response from API
def getResponse(topic):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "physical facts about a " + topic + ", shortest possible answer while maximizing the information, max 500 chars"}
        ]
    )
    
    response = response.choices[0].message.content.strip()
    return response

def predict(imageFile):
    # Read image
    im = cv2.imread(os.path.join("tempAssets", imageFile))

    # Shape image
    image = tf.image.decode_image(tf.io.encode_jpeg(im).numpy(), channels=3)
    resize = tf.image.resize(image, (256, 256))
    
    # Normalize image
    input_image = np.expand_dims(resize / 255.0, axis=0)

    # Make prediction
    prediction = model.predict(input_image, verbose=0)
    predictedClassIndex = np.argmax(prediction)
    predictedClass = classLabels[predictedClassIndex]

    # Feed prediction into API
    return getResponse(predictedClass)