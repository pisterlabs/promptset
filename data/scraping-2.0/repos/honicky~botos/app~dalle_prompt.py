import os
import openai

class DalleClient:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def generate_image(self, prompt):
        return openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )["data"][0]["url"]
