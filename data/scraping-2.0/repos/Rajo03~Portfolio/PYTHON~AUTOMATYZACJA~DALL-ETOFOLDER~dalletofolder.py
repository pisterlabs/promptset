import requests
from io import BytesIO
from PIL import Image
import os
import openai

# Set up API parameters
api_key = "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"
model = "image-alpha-001"
url = "https://api.openai.com/v1/images/generations"

# Prompt user for text input
prompt = input("minimalist print design ")

# Send request to OpenAI API
headers = {"Authorization": f"Bearer {api_key}"}
data = {
    "model": model,
    "prompt": prompt,
    "num_images": 1,
    "size": "512x512",
    "response_format": "url",
}
response = requests.post(url, headers=headers, json=data)

# Check for errors
if response.status_code != 200:
    print("Error generating image.")
    exit()

# Get image URL from response
image_url = response.json()["data"][0]["url"]

# Download image from URL
image_response = requests.get(image_url)
image_data = BytesIO(image_response.content)

# Open image using Pillow library
image = Image.open(image_data)

# Create directory to save image in (if it doesn't already exist)
directory = "S:\\ARCHIWUM\\MINIMALIST SHOP ETSY\\printy"
if not os.path.exists(directory):
    os.makedirs(directory)

# Save image to file
file_name = f"{prompt}.png"
image.save(os.path.join(directory, file_name))
print(f"Image saved to {os.path.join(directory, file_name)}")
print("gotowe")

"TODO: naprawić dlaczego nie przenosi do folderu"