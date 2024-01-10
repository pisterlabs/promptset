import os
import openai

openai.api_key = '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'

response = openai.Image.create(
  prompt="some mice dancing on a table",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

print(image_url)