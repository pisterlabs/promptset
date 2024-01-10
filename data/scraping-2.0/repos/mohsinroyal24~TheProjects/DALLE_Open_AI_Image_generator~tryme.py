import openai
from config2 import key

openai.api_key = key
response = openai.Image.create(prompt="a cricket match", n=1, size="256x256")
image_url = response['data'][0]['url']
print(image_url)
