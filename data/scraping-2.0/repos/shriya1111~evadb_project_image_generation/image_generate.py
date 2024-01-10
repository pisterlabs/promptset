import os
import openai

openai.api_key = os.getenv("OPENAI-KEY")

user_prompt = "city during the night"

response = openai.Image.create(
    prompt = user_prompt, 
    n = 1, 
    size = "1024x1024"
)   

image_url = response['data'][0]["url"]

print(image_url)
