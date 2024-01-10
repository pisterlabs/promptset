# DALL-E based image creator. Change the promt to whatever picture you are looking for.
# 
# To install all required libraries run:
# pip install Pillow openai
# 
# More information on DALL-E and OpenAI:
# https://platform.openai.com/docs/guides/images/introduction

import openai  # OpenAI Python library to make API calls
import requests  # used to download images
import os  # used to access filepaths
import time # for timestamping the filenames

def getImage(api_key, prompt, folder):
    openai.api_key = api_key
    image_dir = os.path.join(os.curdir, folder)
    if not os.path.isdir(image_dir): os.mkdir(image_dir)
    generation_response = openai.Image.create(prompt=prompt, n=1, size="1024x1024", response_format="url",)
    with open(os.path.join(image_dir, str(time.time()) + ".png"), "wb") as image_file:
        image_file.write(requests.get(generation_response["data"][0]["url"]).content)  

# set input variables
api_key = "" # it is a bad idea to put your key in code
folder = "ai/openai/images"
prompt = "Dark pattern of a wolf eating a fish futuristic painting"

# run image creator
getImage(api_key, prompt, folder)
