# This file contains the code for generating an image using openai's davinci engine
# The code is based on the example from https://beta.openai.com/examples/default-example-1

import os
import openai
import logging

# Activate request logging support.
# Caution: Might print the openai key to the console
import http.client as http_client


if os.getenv("DEBUG"):
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
    
    openai.debug = True
    
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        size="512x512",
    )
    return response


def alter_image(prompt, original_image_path, mask_image_path=None):
    response = openai.Image.create_edit(
        prompt=prompt,
        image=open(original_image_path, "rb"),
        mask=open(mask_image_path, "rb") if mask_image_path else None,
        size="256x256",
        n=3
    )
    return response


# Generate an image or a variation of an image
response = generate_image("Context: 8 bit computer game.\r\nTask: Create a gray scale image of"
                          " Luke Skywalker writting source code on a computer terminal")

# Get the absolute path of the image sample.png
# image_path = os.path.abspath("./images/sample.png")
# mask_path = os.path.abspath("./images/mask.png")
# response = alter_image("Add a volcano", image_path, mask_path)

# Display all urls of the image variations
for choice in response["data"]:
    print(f"Image url: {choice['url']}")
