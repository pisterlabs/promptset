import requests
from PIL import Image
from io import BytesIO
import base64
import openai
from dotenv import load_dotenv
import os

load_dotenv()

spotify_client_id = os.environ["SPOTIFY_CLIENT_ID"]
spotify_client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
gpt_key = os.environ["gpt_key"]


def compress_image(url, quality=75):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_format = img.format.lower()
    if img_format == 'jpeg':
        # Already a JPEG, just compress with specified quality level
        img = img.convert('RGB')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=quality)
        img_bytes = img_bytes.getvalue()
    elif img_format in ['bmp', 'png', 'webp']:
        # Convert to JPEG and compress with specified quality level
        img = img.convert('RGB')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG', quality=quality)
        img_bytes = img_bytes.getvalue()
    else:
        # Unsupported image format
        raise ValueError(f"Unsupported image format: {img_format}")
        
    # Convert the image bytes to base64-encoded string
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def get_dalle_image(prompt):
    openai.api_key = gpt_key
    # Define the DALL-E API endpoint
    endpoint = "https://api.openai.com/v1/images/generations"
    # Define the API request data
    data = {
        "model": "image-alpha-001",
        "prompt": prompt,
        "num_images": 1,
        "size": "512x512",
        "response_format": "url"
    }
    response = openai.Image.create(**data)
    # Parse the response to extract the image URL
    image_url = response["data"][0]["url"]
    # Download the image data from the URL
    # image_data = requests.get(image_url).content
    # # Convert the image data to a base64-encoded string
    # image_base64 = base64.b64encode(image_data).decode("utf-8")
    return image_url