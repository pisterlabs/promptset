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

response = client.chat.completions.create(
    #model="gpt-4-vision-preview",
    model="gemini-pro-vision",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image? Please response in Chinese."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://milo-test.oss-cn-zhangjiakou.aliyuncs.com/hdd/batch1/image007.png",
                        "detail": "low"
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.model_dump_json())
