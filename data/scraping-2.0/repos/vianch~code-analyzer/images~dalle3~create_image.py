import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.getenv('OPENAI_API_KEY')

response = openai.images.generate(
    model="dall-e-3",
    prompt="creatte different image of the same pixel art frog to make an animation for game development, the frog pixel art should be 32x32 pixels, add a hat as zelda and a sword",
    size="1024x1024",
    quality="standard",
    n=1,
)

print(response.data[0].url)