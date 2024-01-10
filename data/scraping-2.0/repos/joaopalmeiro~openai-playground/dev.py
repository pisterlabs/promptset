# https://openai.com/blog/dall-e-api-now-available-in-public-beta/
# https://beta.openai.com/docs/libraries/python-bindings
# https://beta.openai.com/docs/guides/images
# https://labs.openai.com/
# https://beta.openai.com/docs/api-reference/images/create
# https://github.com/un33k/python-slugify
# https://beta.openai.com/examples

import os
import urllib.request
from datetime import datetime

import openai
from dotenv import load_dotenv
from slugify import slugify

IMAGE_SIZES = {
    "S": "256x256",
    "M": "512x512",
    "L": "1024x1024",
}

if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = "Pikachu painted by El Greco."

    response = openai.Image.create(prompt=prompt, n=1, size=IMAGE_SIZES["L"])
    image = response["data"][0]
    image_url = image["url"]

    print(image_url)
    urllib.request.urlretrieve(
        image_url,
        f"output/{slugify(prompt)}_openai_{int(datetime.now().timestamp())}.png",
    )
