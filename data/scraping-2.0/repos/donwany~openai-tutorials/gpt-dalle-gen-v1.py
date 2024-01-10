from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class ImageGenerator:
    def __init__(self, api_key: str, model: str = "dall-e-3"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_image(self, prompt, size="1024x1024", quality="standard", n=1):
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n
        )
        return response.data[0].url


def main():
    api_key = os.getenv("api_key")
    image_generator = ImageGenerator(api_key)

    prompt = "a white siamese cat sitting in a spaceship"
    image_url = image_generator.generate_image(prompt)

    print(image_url)


if __name__ == '__main__':
    main()
