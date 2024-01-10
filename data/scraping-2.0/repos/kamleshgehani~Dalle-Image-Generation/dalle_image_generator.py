import os
import openai
import requests
from PIL import Image
from io import BytesIO

# Set your OpenAI API key
openai.api_key = "Insert Your Own API Key"

def generate_images(prompt, output_path, n=1):
    # Call DALL-E API
    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size="256x256",
        model="image-alpha-001"
    )

    # Download and save the generated images
    for idx, data in enumerate(response['data']):
        image_url = data['url']
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(output_path, f"{prompt}_{idx}.png"))

if __name__ == "__main__":
    prompt = input("Enter the text prompt to generate images: ")
    output_path = input("Enter the output folder path: ")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generate_images(prompt, output_path)
    print(f"Image saved in {output_path}")