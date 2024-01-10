import openai
import requests
from requests.structures import CaseInsensitiveDict

import json

openai.api_key = "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"


def generate_image(prompt, model, api_key):
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = f"Bearer {api_key}"

    data = """
    {
        """
    data += f'"model": "{model}",'
    data += f'"prompt": "{prompt}",'
    data += """
        "num_images":1,
        "size":"1024x1024",
        "response_format":"url"
    }
    """

    resp = requests.post("https://api.openai.com/v1/images/generations", headers=headers, data=data)

    if resp.status_code != 200:
        raise ValueError("Failed to generate image")

    response_text = json.loads(resp.text)
    return response_text['data'][0]['url']


# 生成一个带眼镜的亚洲女性头像
prompt = "An Asian woman with glasses"

url = generate_image(prompt, "image-alpha-001", openai.api_key)
response = requests.get(url)
with open("generated_image.jpg", "wb") as f:
    f.write(response.content)
