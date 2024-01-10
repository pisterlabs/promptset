import openai
from env import OpenAIKeys

def generate_image(description):
    """Generate an image based on a given description."""
    # Making an API call to generate an image
    # API_ENDPOINT would be the URL where you can POST the data to generate an image
    # API_KEY would be your authentication key
    openai.api_key = OpenAIKeys.OPENAI_API_KEY
    

    response = openai.Image.create(
    model="dall-e-3",
    prompt=description,
    size="1024x1024",
    quality="standard",
    n=1,
    )

    return response['data'][0]['url']
