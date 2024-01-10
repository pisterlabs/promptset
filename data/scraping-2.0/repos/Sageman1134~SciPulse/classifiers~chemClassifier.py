# Import libraries
import openai
from dotenv import load_dotenv
import os
import tensorflow as tf
import numpy as np
import cv2

# Define class labels
classLabels = ["perfectly titrated", "overly titrated", "under titrated"]

# Load trained model
model = tf.keras.models.load_model("classifiers/chemModel")

# Define API key
load_dotenv("Vars.env")
key = os.getenv("APIKEY")
openai.api_key = key

# Get response from API
def getResponse(topic):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "tell me about a " + topic + ", and how it might be improved, if possible. do it very short in 2 sentences max 500 char."}
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