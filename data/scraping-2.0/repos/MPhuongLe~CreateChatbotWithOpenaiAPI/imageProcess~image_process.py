import openai
import requests
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_image(prompt,size="512x512"):
    response = openai.Image.create(
        prompt=prompt,
        #model=model,
        n = 1,
        size=size
    )

    # Get the image URL from the API response
    image_url = response['data'][0]['url']

    # Download the image and convert it to a PIL Image object
    # image_data = requests.get(image_url).content
    # image = Image.open(BytesIO(image_data))

    return image_url

if __name__ == '__main__':
    prompt = "a cuisine with egg and milk"
    image = generate_image(prompt, size="512x512")
    image.show()
