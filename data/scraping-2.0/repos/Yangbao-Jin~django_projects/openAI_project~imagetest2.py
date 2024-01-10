import openai
import os


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv('OPENAI_API_KEY')

import base64
import requests

# OpenAI API Key
#api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "recur1.png"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      { 
        "role": "system",
        "content": 
          {
            "type": "text",
            "text": "你是一个递归函数计算专家，这是一个数学的递归表达式,请识别图像中的递归表达式，然后回答图片上的问题"
          },
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这是一个递归问题,请识别图片上的递归表达式，如果图片上有问题，请回答图片上的问题"
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
    "max_tokens": 600
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json=response.json()

content = response_json['choices'][0]['message']['content']
print(content)