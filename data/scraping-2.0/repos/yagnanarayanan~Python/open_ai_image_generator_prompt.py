# Generate an image using OpenAI API by providing a prompt and image size
import openai
import os
import requests
import time
# provide the openai API key
openai.api_key = os.getenv('OPENAI_API_KEY')
# get the prompt from user
prompt = input("Please enter a prompt for which you need an image generated using OpenAI: \n")
# get the image size from user
size = int(input("Please enter the size of the image: "))
# get an image as a response from openai
res = openai.Image.create(prompt=prompt, n=1, size=f"{size}x{size}")
# pick the URL from the response
url = res["data"][0]["url"]
# hit the url using a request
response = requests.get(url)
# write the image from the response content
with open(f"image_{int(time.time())}.jpeg", 'wb') as f:
    f.write(response.content)
