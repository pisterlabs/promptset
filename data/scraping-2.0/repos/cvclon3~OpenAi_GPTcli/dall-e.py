import openai
import requests

with open("key.txt", "r") as k:
    key = k.readline()

openai.api_key = key

response = openai.Image.create(
  prompt="A cute baby anime girl",
  n=2,
  size="1024x1024"
)

image_url = response["data"][0]["url"]

with open(f"./images/image{hash}", "wb") as img: