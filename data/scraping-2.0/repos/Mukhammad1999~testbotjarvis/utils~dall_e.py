import openai
import requests
import base64
import io
from data.config import OPENAI_API

openai.api_key = OPENAI_API


async def generate_image(prompt):
        # Отправляем текст в DALL-E для генерации изображения
    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        },
        json={
            "prompt": prompt,
            "num_images": 1,
            "size": "512x512",
            "response_format": "b64_json",
        },
    )
    if response.json() != None:
            
        image_data = response.json()['data'][0]['b64_json']
        image_binary = base64.b64decode(image_data)
        image_file = io.BytesIO(image_binary)
        return image_file
    else:
        print('error')
        return None