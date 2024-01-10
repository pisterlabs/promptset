# encoding: UTF-8
import os
import openai
import json

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

import requests
import base64
from PIL import Image
from io import BytesIO

def get_image_as_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Determine the image type
        image = Image.open(BytesIO(response.content))
        image_type = image.format
        if image_type:
            prefix = f"data:image/{image_type.lower()};base64,"
            return prefix + base64.b64encode(response.content).decode('utf-8')
        else:
            raise Exception("Unable to determine the image type.")
    else:
        raise Exception("Failed to download the image.")

url = "https://images.unsplash.com/5/unsplash-kitsune-3.jpg?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxjb2xsZWN0aW9uLXBhZ2V8M3wxMDA0NjU5fHxlbnwwfHx8fHw%3D"

image_data = get_image_as_base64(url)

print(image_data[:100])

response = client.chat.completions.create(
    #model="gpt-4-vision-preview",
    model="gemini-pro-vision",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this image? Please response in Chinese."},
            {"type": "image_url", "image_url": {"url": image_data,"detail": "low"}},
        ],
    }],
    max_tokens=300,
)
print(response.model_dump_json())
