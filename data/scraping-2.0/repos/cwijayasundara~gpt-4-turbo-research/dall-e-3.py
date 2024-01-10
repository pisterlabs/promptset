import openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
from IPython.display import display

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

print(openai.api_key)

PROMPT = ("Commercial LLMs like GPT-4, Opensource LLMs like LLaMA2 and NLP transformer models like T5 fighting for "
          "supremacy.")

client = OpenAI()

response = client.images.generate(
    model="dall-e-3",
    prompt=PROMPT,
    size="1024x1024",
    quality="standard",
    n=1,
)

image_url = response.data[0].url
print(image_url)


def display_and_save_image_from_url(url, scale_percent=100, save_name='image.png'):
    # Send a GET request to the specified URL to retrieve the image
    response = requests.get(url)
    # Open the image
    img = Image.open(BytesIO(response.content))

    # Calculate the new size, as a percentage of the original size
    if scale_percent != 100:
        width, height = img.size
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        img = img.resize((new_width, new_height))

    # Save the image locally with the given name
    img.save(save_name)

    # Display the image in the notebook
    display(img)


# Call the function with the URL, the scale percentage, and the save name you want
display_and_save_image_from_url(image_url, scale_percent=50, save_name='llms.png')
