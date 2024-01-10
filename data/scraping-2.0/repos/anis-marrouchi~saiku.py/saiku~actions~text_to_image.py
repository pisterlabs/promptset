import os
import requests
from openai import OpenAI
from pathlib import Path

class TextToImageAction:
    def __init__(self, agent):
        self.agent = agent
        self.name = 'text_to_image'
        self.description = 'Generates an image based on a description using OpenAI\'s DALL-E 3 model, saves it to disk, and returns the filename.'
        self.parameters = [{
            'name': 'description',
            'type': 'string',
            'required': True,
            'description': self.description
        }]
        dependencies = ['openai', 'requests', 'pathlib']

    async def run(self, description):
        openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        response = await openai.images.generate(
            model="dall-e-3",
            prompt=description,
            n=1,
            size="1024x1024"
        )

        if not response.data:
            return 'Image generation failed'

        image_url = response.data[0].url
        if not image_url:
            return 'Image generation failed'

        # Download and save the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            # Define the path where you want to save the image
            image_path = Path("generated_images") / f"{description.replace(' ', '_')}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as file:
                file.write(image_response.content)
            
            return f'Image saved to {image_path}'
        else:
            return 'Failed to download the image'