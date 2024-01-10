import base64
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


async def describe_image(agent, image_path: str, prompt: str = "Describe this image", detail: str = "auto"):
    image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}",
                            "detail": f"{detail}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    answer = response.json()
    return answer['choices'][0]['message']['content']


async def describe_images(agent, image_paths: list[str], prompt: str = "Describe this image", detail: str = "auto"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    }

    content = [{
        "type": "text",
        "text": f"{prompt}"
    },]
    for path in image_paths:
        image = encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
                "detail": f"{detail}"
            }
        })
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 1000
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    answer = response.json()
    return answer['choices'][0]['message']['content']

tool_describe_image = {
    "info": {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": "Describes an image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image to describe, relative to the current working directory"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to use for the image description, defaults to 'Describe this image', can be anything else as well, like: 'how would you solve this puzzle?'"
                    },
                    "detail": {
                        "type": "string",
                        "description": "Level of detail for image processing, 'low' for fast processing, 'high' for more detailed processing, 'auto' for automatic selection of detail level, defaults to 'auto'",
                        "enum": ["auto", "low", "high"]
                    }
                },
                "required": ["image_path"],
            },
        }
    },
    "function": describe_image,
}

tool_describe_images = {
    "info": {
        "type": "function",
        "function": {
            "name": "describe_images",
            "description": "Describes images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_paths": {
                        "type": "array",
                        "description": "Paths to images to describe, relative to the current working directory"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to use for the image description, defaults to 'Describe this image', can be anything else as well, like: 'what am I doing?'"
                    },
                    "detail": {
                        "type": "string",
                        "description": "Level of detail for image processing, 'low' for fast processing, 'high' for more detailed processing, 'auto' for automatic selection of detail level, defaults to 'auto'",
                        "enum": ["auto", "low", "high"]
                    }
                },
                "required": ["image_paths"],
            },
        }
    },
    "function": describe_images,
}
