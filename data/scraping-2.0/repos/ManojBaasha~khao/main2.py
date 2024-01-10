import io
import os
import re
import numpy as np
import requests
from PIL import Image
from google.cloud import vision
import openai

# Set up Google Vision API client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./application_default_credentials.json"
client = vision.ImageAnnotatorClient()

# Set up OpenAI GPT-3.5 model
openai.api_key = "ysk-w169lBM2inicPbAVz0xyT3BlbkFJpFxPOQYslCgmYAtE6EcH"
model_name = "gpt-3.5-turbo"

image_path = "Flyer-Instagram.png"

# Loads the image into memory
with io.open(image_path, 'rb') as image_file:
    content = image_file.read()
    
image = types.Image(content=content)
# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations
print('Labels:')
for label in labels:
    print(label.description)