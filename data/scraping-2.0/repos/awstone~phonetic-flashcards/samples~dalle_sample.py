import openai
import os 
from dotenv import load_dotenv
load_dotenv('/home/awstone/.bashrc')  # take environment variables from my local .bashrc

openai.api_key = os.environ["OPENAI_API_KEY"]
response = openai.Image.create(
  prompt="a white siamese cat",
  n=1,
  size="256x256"
)
image_url = response['data'][0]['url']

print(image_url)