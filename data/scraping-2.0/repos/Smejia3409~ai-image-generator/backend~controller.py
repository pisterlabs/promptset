import os
import openai

# gets api key from env variable
openai.api_key = os.getenv("OPEN_AI_KEY")

def create_image(propmt):
    response = openai.Image.create(
        prompt=propmt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    return image_url