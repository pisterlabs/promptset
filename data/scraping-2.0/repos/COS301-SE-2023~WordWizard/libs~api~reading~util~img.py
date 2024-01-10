import os
from dotenv import load_dotenv

load_dotenv()
import openai

openai.api_key = os.getenv("OPEN_AI_KEY")


def generate_image(prmpt: str):
    response = openai.Image.create(
        prompt=prmpt,
        n=1,
        size="256x256"
    )
    image_url = response['data'][0]['url']
    return image_url