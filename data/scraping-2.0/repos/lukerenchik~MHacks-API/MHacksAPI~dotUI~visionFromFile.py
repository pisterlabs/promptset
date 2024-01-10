import base64

import requests
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="variables.env")
apiKey = os.getenv('OPENAI_API_KEY')

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {apiKey}"
}

def describe_image(image_path):
    """
    This function takes an image URL and returns a description of the image using OpenAI's API.

    Args:
    image_url (str): The URL of the image to be described.

    Returns:
    str: A description of the image.
    """

    apiKey = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=apiKey)
    prompt = "Describe the core aspects of the image using a single sentence designed at giving a user who has never seen the image a clear idea of what is happening."
    payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
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
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    completion_content = response.json()['choices'][0]['message']['content']

    return completion_content

#output = describe_image(base64_image)
#print(output)