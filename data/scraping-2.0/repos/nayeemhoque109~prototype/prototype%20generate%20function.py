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

num_frames = 5

# Define a list of prompts
prompts = [
    "Subject: Full moon.",
    "Subject: Full moon with in the sky with stars.",
    "Subject: Full moon with in the sky with stars and skyline.",
    "Subject: Full moon with in the sky with stars and skyline from a painting in a room.",
    "Subject: Full moon and a little house.",
]

# Initialize the list before the loop
image_objects = []

# Create a variation for each image in the list
for i in range(num_frames):

    # Use a different prompt for each image
    prompt = prompts[i]

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

    # Save each image and print a message
    for i, data in enumerate(image_data_list):
        img = Image.open(BytesIO(base64.b64decode(data)))
        img.save(f"{img_filename}_{i}.png")
        print(f"{img_filename}_{i}.png was saved")
        image_objects.append(img)

# Create a GIF from the saved images
gif_filename = f"{img_filename}_variations.gif"
imageio.mimsave(gif_filename, [img for img in image_objects], duration=0.5)

print(f"GIF file {gif_filename} was saved")

# Create a tkinter window
window = tk.Tk()

# Create a photo image from the GIF
photo_image = tk.PhotoImage(file=gif_filename)

# Create a label with the photo image
label = tk.Label(window, image=photo_image)

# Pack the label
label.pack()

# Start the tkinter main loop
window.mainloop()


   
