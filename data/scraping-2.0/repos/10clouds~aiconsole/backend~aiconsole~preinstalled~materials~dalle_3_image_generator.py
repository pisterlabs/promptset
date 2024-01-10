import asyncio
import os
from typing import Literal
from urllib.parse import unquote, urlparse

import openai
import requests
from pydantic import BaseModel

from aiconsole.core.settings.project_settings import Settings


class Image(BaseModel):
    relative_path: str
    revised_prompt: str


def _extract_filename_from_url(url):
    """
    Extracts the filename from a URL, handling cases where query parameters contain '/'.

    :param url: The URL to extract the filename from.
    :return: The extracted filename, or None if no filename is found.
    """
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the path component of the URL
    path = parsed_url.path

    # Decode URL encoding
    decoded_path = unquote(path)

    # Extract the filename
    filename = decoded_path.split("/")[-1] if "/" in decoded_path else decoded_path

    return filename


def generate_image(prompt: str, size: Literal["1024x1024", "1792x1024", "1024x1792"]) -> list[Image]:
    """
    Returns a list of:

    class Image(BaseModel):
      relative_path: str
      revised_prompt: str
    """
    settings = Settings()
    asyncio.run(settings.reload())
    settings.stop()

    openai_key = settings.get_openai_api_key()

    client = openai.OpenAI(api_key=openai_key)

    # Call the API
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )

    ret = []

    # download the image to a file in cwd
    for i, image in enumerate(response.data):
        image_url = image.url
        if not image_url:
            raise Exception("No image url returned")
        image_name = _extract_filename_from_url(image_url)

        image_path = os.path.join("images", image_name)
        os.makedirs(os.path.join("images"), exist_ok=True)
        r = requests.get(image_url, allow_redirects=True)
        open(image_path, "wb").write(r.content)
        ret.append(Image(relative_path=image_path, revised_prompt=image.revised_prompt or ""))

    return ret
