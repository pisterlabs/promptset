import os
from hashlib import sha256
import aiohttp
from mensautils.parser.canteen_result import Serving
import logging
import openai
import asyncio
from lunchbot.config import PUBLIC_URL

openai.api_key = os.getenv("OPENAI_KEY")

LOGGER = logging.getLogger(__name__)


async def generate_preview_image(title: str) -> bytes:
    LOGGER.info(f"Requesting a preview for '{title}'")
    loop = asyncio.get_running_loop()
    dalle_resp = await loop.run_in_executor(
        None, lambda: openai.Image.create(prompt=title, n=1, size="512x512")
    )
    image_url = dalle_resp.data[0]["url"]
    LOGGER.info(f"Image url: {image_url}")
    async with aiohttp.ClientSession() as session:
        response = await session.get(image_url)
        assert response.status == 200
        LOGGER.info(f"Generated a preview for '{title}'")
        return await response.read()


def publish_image(prefix: str, folder: str, image: bytes) -> str:
    name = f"{prefix}.png"
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        f.write(image)
    return name


async def get_preview_image_path(title: str, folder: str) -> str:
    titlehash = sha256(title.encode(), usedforsecurity=False).hexdigest()[:20]
    filename = f"{titlehash}.png"
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        image = await generate_preview_image(title)
        publish_image(titlehash, folder, image)

    return filename


async def create_menu_images(menu: list[Serving], folder: str) -> list[str]:
    urls = []
    for item in menu:
        image_path = await get_preview_image_path(item.title, folder)
        urls.append(f"{PUBLIC_URL}/{image_path}")
    return urls
