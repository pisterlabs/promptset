import os
import openai

openai.api_key = "API-KEY-here"

response = openai.Image.create(
  prompt="space unicorn",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

print(image_url)