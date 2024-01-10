import base64
from openai import OpenAI


class ImageGenerator:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client

    def generate(self, prompt):
        response = self.openai_client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
            quality="standard",
        )
        image = base64.b64decode(response.data[0].b64_json)
        return image
