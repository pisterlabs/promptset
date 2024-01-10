import openai
import requests
from PIL import Image


openai.api_key_path = 'openai_api_key.txt'


def generate_text2img(text, size = '256x256'):

    if size not in ['256x256', '512x512', '1024x1024']:
        raise ValueError("Invalid size. Size must be one of ['256x256', '512x512', '1024x1024'].")  
    prompt = "Generates the main image of the company's website based on this description: " + text
    res = openai.Image.create(prompt = prompt, n = 1, size = size)

    url = res['data'][0]['url']

    response = requests.get(url)

    Image.open(response.raw)