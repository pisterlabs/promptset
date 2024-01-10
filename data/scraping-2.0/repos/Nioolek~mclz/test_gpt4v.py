import os

import openai
import requests
import time
import json
import time

API_SECRET_KEY = "sk-sU0lrtfAsUnEMtHgeAWN1I4lEuQxpb1OHwnUXZSRElsB2n26";
BASE_URL = "https://api.chatanywhere.com.cn/v1/"  # 智增增的base_url

from openai import OpenAI


# gpt4v
def gpt4v(query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    print(resp)
    print(resp.choices[0].message.content)


if __name__ == '__main__':
    gpt4v("What are in these images? Is there any difference between them?");