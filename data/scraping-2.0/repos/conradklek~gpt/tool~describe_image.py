""" This file contains the AI functions for the AI module """

import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def encode_image(image_path: str):
    """
    Encode an image to base64
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def describe_image(image_path: str):
    """
    Describe an image with DALL-E
    """
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """\
Please describe the image below in great detail. \
Describe the objects, the colors, the shapes, the textures, and the materials. \
Describe the lighting and the shadows. Describe the scene and the composition. \
Describe the emotions and the feelings that the image evokes. \
Describe the image as if you were describing it to a blind person."""
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
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    response_str = response.json()["choices"][0]["message"]["content"]
    return response_str


available_functions = {
    "describe_image": describe_image
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Returns a description of an image from a path",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The path to the image to describe"
                    }
                },
                "required": ["image_path"]
            }
        }
    }
]
