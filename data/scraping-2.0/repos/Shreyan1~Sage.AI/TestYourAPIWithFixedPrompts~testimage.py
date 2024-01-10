import openai
from apikey import APIKEY

openai.api_key = APIKEY 

response = openai.Image.create(
  model="dall-e-3",
  prompt="Adam and Eve",
  size="1024x1024",
  quality="standard",
  n=1,
)

print(response.data[0].url)