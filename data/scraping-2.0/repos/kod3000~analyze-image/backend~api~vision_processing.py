import os
import requests
from .constants import OPENAI_SAMPLE_RESPONSE as sample

correct_api_key = "sk-" + os.environ.get('OPENAI_API_KEY_1')
wrong_api_key = "sk-" + os.environ.get('OPENAI_API_KEY_2')

def process_vision(base64_image, use_correct_key=True):
    """
    Process the image using OpenAI's API

    Takes in a base64 encoded image and returns the response from OpenAI

    Developer Notes :
    - You can pass in false flag to test error handling
    """
    api_key = wrong_api_key  # Default to wrong key
    if use_correct_key:
        api_key = correct_api_key
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # return sample # For testing
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response)
    print(response.json())
    return response.json()
