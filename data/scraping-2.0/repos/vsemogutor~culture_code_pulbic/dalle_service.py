import requests
from openai import OpenAI
import uuid
from image import Image

def download_image(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Return the binary content of the response
        return response.content
    except requests.RequestException as e:
        # Handle any exceptions (like network issues or invalid URLs)
        print(f"Error occurred: {e}")
        return None
        
class DalleService:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_image(self, prompt):
        client = OpenAI(api_key = self.api_key)

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            )

        image_url = response.data[0].url
        image_data = download_image(image_url)
        image_name = f"{uuid.uuid4()}.jpg"
        return Image(image_name, image_data)


