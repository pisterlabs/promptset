import openai
import requests
from pprint import pprint

from config import API_KEY

from bot_db import engine, session

# Setup openai 
openai.api_key = API_KEY

pprint(openai)

class DallEClient:
    """
        Initialize DallEClient
            configure api_key
    """
    def __init__(self, API_KEY) -> None:
        self.API_KEY = API_KEY
        openai.api_key = self.API_KEY

    def save_image(self, url, name) -> str:
        response = requests.get(url)
        import os 
        current_path = os.getcwd()
        print("Current path: ", current_path)
        image = open(f"images/{name}.jpg", "wb").write(response.content)
        return f"{current_path}/images/{name}.jpg"

    def create_sample(self, prompt, number, size):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=number,
                size=size
            )
            image_url = response['data'][0]['url']
            image_path = self.save_image(url=image_url, name=prompt)
            return image_path
        except Exception as e:
            print("Exception: ", e)
            raise Exception("Error: ", e)
        

