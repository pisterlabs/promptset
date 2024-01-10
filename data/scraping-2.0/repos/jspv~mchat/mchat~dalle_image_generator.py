"""Utility that calls OpenAI's Dall-E Image Generator."""
from typing import Any, Dict, Optional
from openai import OpenAI
from functools import partial
import asyncio


class DallEAPIWrapper(object):
    """Wrapper for OpenAI's DALL-E Image Generator.

    https://platform.openai.com/docs/guides/images/generations?context=node

    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        num_images: int = 1,
        size: str = "1024x1024",
        separator: str = "\n",
        model: Optional[str] = "dall-e-3",
        quality: Optional[str] = "standard",
    ):
        self.openai_api_key = openai_api_key
        self.num_images = num_images
        """Number of images to generate"""

        self.size = size
        """Size of image to generate"""

        self.separator = separator
        """Separator to use when multiple URLs are returned."""

        self.model = model
        """Model to use for image generation"""

        self.quality = quality
        """Quality of the image that will be generated"""

        self.client = OpenAI(api_key=self.openai_api_key)

    def run(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        response = self.client.images.generate(
            prompt=query,
            n=self.num_images,
            size=self.size,
            model=self.model,
            quality=self.quality,
        )
        image_urls = self.separator.join([item.url for item in response.data])
        return image_urls if image_urls else "No image was generated"

    async def arun(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        loop = asyncio.get_running_loop()
        # prepare keyword arguments for the run_in_executor call by partial
        sync_func = partial(
            self.client.images.generate,
            prompt=query,
            n=self.num_images,
            size=self.size,
            model=self.model,
            quality=self.quality,
        )
        response = await loop.run_in_executor(None, sync_func)
        image_urls = self.separator.join([item.url for item in response.data])
        return image_urls if image_urls else "No image was generated"
