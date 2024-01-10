# create.py

import os

import openai
import requests


PROMPT = "An eco-friendly computer from the 90s in the style of vaporwave"
#Choose size of image
SIZE = "500x500"

# Load .env file if it exists
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size=SIZE,
    
)

print(response["data"][0]["url"])

image_url = response["data"][0]["url"]

# Then, download the image from the URL and save it to a file
response = requests.get(image_url)

with open("images/eco-friendly-computer.png", "wb") as f:
    f.write(response.content)