import json
from dotenv import load_dotenv
import requests
import os
from openai import OpenAI


load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="This is an image of a wooden boardwalk extending through a lush green meadow under a beautiful blue sky with scattered clouds. The boardwalk provides a path through the grass, likely intended for walking while preserving the natural environment. To the right and left of the boardwalk, tall grasses dominate the landscape. In the distance, there are trees and shrubs dotting the horizon, further adding to the natural, serene setting. The lighting suggests it may be late afternoon or early evening, with the sun providing a warm, soft light over the landscape.",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)
