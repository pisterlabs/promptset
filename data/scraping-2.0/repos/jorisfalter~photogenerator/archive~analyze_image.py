from openai import OpenAI
import base64
import requests
import os
import sys
import json
from dotenv import load_dotenv
import sys
import base64

# Load .env file
load_dotenv()

# OpenAI API Key
api_key = os.getenv('API_KEY')

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# print("we're inside python 1")

# Function to encode the image


# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# start temp storage

def encode_image_from_stdin():
    return base64.b64encode(sys.stdin.buffer.read()).decode('utf-8')


# Getting the base64 string from stdin
base64_image = encode_image_from_stdin()
# end temp storage

# Accept image path from command line argument
# The first argument is the script name, so we take the second argument.
# image_path = sys.argv[1]

# # Getting the base64 string
# base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 300
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

# Extract the description from the analysis response
# This assumes the response format includes a text description in the expected part of the response JSON
description_input = response.json()['choices'][0]['message']['content']
description_fixed = "the following is a description of a drawing made by a child, I would like you to turn it into a photo realistic image, suitable for children: "
description = description_fixed + description_input


# Image generation payload
generation_payload = {
    "model": "dall-e-3",
    "prompt": description,
    "size": "1024x1024",
    "quality": "standard",
    "n": 1,
}

# Send the image generation request
generation_response = client.images.generate(**generation_payload)

# Extract the image URL from the generation response
image_url = generation_response.data[0].url

# print(f"Image URL: {image_url}")
# print(f"Description: {description}")


output = {
    "image_url": image_url,
    "description": description
}
print(json.dumps(output))
