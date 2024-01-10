import openai
import os

req = input("What do you wanna see ? ")

openai.api_key = os.getenv('OPENAI_KEY') 
response = openai.Image.create(
  prompt=req,
  n=1,
  size="256x256"
)
image_url = response['data'][0]['url']
print(image_url)
