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
image_path = "dc4.png"

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
            "text": "这是一个数字电路图,图像左边的都是输入，输出在右侧，有时没有变量说明，需要你自己加上个。图像的符号都遵循数字电路与或非门的标准"
          },
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "这是一个数字电路图,识别这个数字电路图，输出电路的布尔逻辑表达式，不用化简。如果图片上有问题，请回答图片上的问题"
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
    "max_tokens": 1600
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response_json=response.json()

content = response_json['choices'][0]['message']['content']
print(content)