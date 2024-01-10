'''
requires:
pip install openai
pip install --upgrade openai
pip install pillow
pip install imageIO
'''
import os
from io import BytesIO
import imageio
import openai                  
from datetime import datetime  
import base64                 
import requests                
from PIL import Image          
import tkinter as tk           
from PIL import ImageTk        
import requests
import imageio.v2 as imageio

def old_package(version, minimum):  
    version_parts = list(map(int, version.split(".")))
    minimum_parts = list(map(int, minimum.split(".")))
    return version_parts < minimum_parts

if old_package(openai.__version__, "1.2.3"):
    raise ValueError(f"Error: OpenAI version {openai.__version__}"
                     " is less than the minimum version 1.2.3\n\n"
                     ">>You should run 'pip install --upgrade openai')")

from openai import OpenAI

# Set your OpenAI API key
api_key = "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"

# Create the OpenAI client
client = OpenAI(api_key=api_key)

prompt = (
 "Subject: planets. " 
    "Style: cartoon."    
)

image_params = {
    "model": "dall-e-2",
    "n": 1,
    "size": "256x256",
    "prompt": prompt,
    "user": "myName",
    "response_format": "b64_json"
}

try:
    images_response = client.images.generate(**image_params)
except Exception as e:
    print(f"An error occurred: {e}")
    raise

images_dt = datetime.utcfromtimestamp(images_response.created)
img_filename = images_dt.strftime('DALLE-%Y%m%d_%H%M%S')

revised_prompt = images_response.data[0].revised_prompt

image_data_list = [image.model_dump()["b64_json"] for image in images_response.data]

image_objects = []

if image_data_list and all(image_data_list):
    for i, data in enumerate(image_data_list):
        image_objects.append(Image.open(BytesIO(base64.b64decode(data))))
        image_objects[i].save(f"{img_filename}_{i}.png")
        print(f"{img_filename}_{i}.png was saved")
else:
    print("No image data was obtained. Maybe bad code?")
    
# Convert "b64_json" data to png file
for i, data in enumerate(image_data_list):
    image = Image.open(BytesIO(base64.b64decode(data)))  # Open the image
    image_objects.append(image)  # Append the Image object to the list

    # Resize the image
    width, height = 256, 256
    image = image.resize((width, height))

    # Convert the image to a BytesIO object
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')

# Set byte_array to the data of the last image generated
byte_array = byte_stream.getvalue()

# Initialize an empty list to store the URLs of the image variations
num_frames = 4
urls = []

# Create a variation for each frame in the animation
for i in range(num_frames):
    try:
        image_params = {
            "image": byte_array,
            "n": 1,
            "model": "dall-e-2",
            "size": "256x256",
            "response_format": "url"
        }

        # Make the request to the API
        images_response = client.images.create_variation(**image_params)

        # Get the URL of the image
        url = images_response.data[0].url
        urls.append(url)

    except openai.OpenAIError as e:
        print(e.http_status)
        print(e.error)

# Download each image from its URL and save it to a local file
for i, url in enumerate(urls):
    response = requests.get(url)
    filename = f"{img_filename}_{i+1}_variation.png"  # Use i+1 to generate a unique filename for each image
    with open(filename, 'wb') as f:
        f.write(response.content)
        print(f"{img_filename}_{i+1}_variation.png was saved")

    # Open the downloaded image and append it to image_objects
    img = Image.open(filename)
    image_objects.append(img)

# Create a GIF from the saved images
gif_filename = f"{img_filename}_variations.gif"
# Adjust the duration to 0.5 seconds for a smoother animation
imageio.mimsave(gif_filename, [img for img in image_objects], duration=0.5)

print(f"GIF file {gif_filename} was saved")
   


   
