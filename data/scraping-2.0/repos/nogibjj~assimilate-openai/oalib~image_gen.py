"""A library for generating images using OpenAI."""

import openai

# build a function that takes a prompt and returns a generated image
def generate_image(prompt, size="1024x1024"):
    """Generate an image using OpenAI's API."""
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size=size,
    )
    image_url = response["data"][0]["url"]
    return image_url


