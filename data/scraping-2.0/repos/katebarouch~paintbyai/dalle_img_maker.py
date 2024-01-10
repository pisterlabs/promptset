from dotenv import load_dotenv
import os
load_dotenv("secrets.sh")
import openai
import requests
from PIL import Image
import io

def dalle(prompt):
    # Define OpenAI key
    api_key = os.getenv("OPENAI_API_KEY")
    print("API KEY = ", api_key)
    openai.api_key = api_key

    # Generate an image
    response = openai.Image.create(
        prompt=prompt,
        size="1024x1024",
        response_format="url"
    )
    print(response)
    return response

def generate_image(api_url, painting_id):
    # Extract the URL from the response dictionary
    url = api_url['data'][0]['url']

    # Make a request to the DALL·E API to get the image data
    response = requests.get(url)
    response.raise_for_status()
    image_data = response.content

    # Load the image data into a Pillow image object
    image = Image.open(io.BytesIO(image_data))

    # Save the image as a JPEG file
    img_path = f'static/images/{painting_id}dalle.jpg'
    image.save(img_path, 'JPEG')

    return img_path

def make_dalle_img(prompt, painting_id):
    response = dalle(prompt)
    img_path = generate_image(response, painting_id)

    return img_path
