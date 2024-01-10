from openai import OpenAI
from download import download_image
from input import get_input

# enter your API key here
api_key='YOUR_API_KEY_HERE'
client = OpenAI(api_key=api_key)


# generates an image from prompt with specified quality
def generate_image(prompt, quality):
    print("Generating Image...")
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality=quality,
        n=1,
    )
    print("Image Generated!")
    image_url = response.data[0].url
    return image_url