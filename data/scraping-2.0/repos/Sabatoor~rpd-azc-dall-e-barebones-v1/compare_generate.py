import openai
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_image(prompt):
    # Read the image file from disk and resize it
    image = Image.open("car.jpeg")
    width, height = 1024, 1024
    image = image.resize((width, height))

    # Convert the image to a BytesIO object
    byte_stream = BytesIO()
    image.save(byte_stream, format='PNG')
    byte_array = byte_stream.getvalue()

    # Generate an image with OpenAI API
    response = openai.Image.create_variation(
        image=byte_array,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    image_response = requests.get(image_url)
    image_data = BytesIO(image_response.content)
    return image_data

if __name__ == '__main__':
    filename = 'generated_image.png'
    i = 1
    while os.path.isfile(filename):
        # File already exists, add suffix to filename
        i += 1
        filename = f"generated_image_{i:03d}.png"
    image_data = generate_image('add a person to a red corvette car')
    with open(filename, 'wb') as f:
        f.write(image_data.read())
