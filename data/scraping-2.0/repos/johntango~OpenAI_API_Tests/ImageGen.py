import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

print("Welcome to ImageGen")

response = client.images.generate(
  model="dall-e-3",
  prompt="A high def image of train",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url

print(image_url)