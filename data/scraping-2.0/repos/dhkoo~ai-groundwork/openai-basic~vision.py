import base64
import requests
import openai
import os

from dotenv import load_dotenv
from utils import read_prompt_template

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


persona_prompt = read_prompt_template("../prompt_templates/arit.txt")
image_path = "../assets/image.jpeg"
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai.api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{persona_prompt}의 페르소나를 기반으로 이 이미지에 대해 설명해줘."
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
    "max_tokens": 500
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

response_json = response.json()
content = response_json['choices'][0]['message']['content']
print(content)

