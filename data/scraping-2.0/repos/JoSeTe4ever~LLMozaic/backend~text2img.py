import dotenv
import os
import openai

dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY");

def transform(prompt):
    response = openai.Image.create(
      prompt=prompt,
      n=1,
      size="512x512"
    )
    image_url = response['data'][0]
    return image_url