"""
To determine how many tokens a given image will use when passed to GPT-4V, let's do a naive binary search for the
resolutions of an image which change the number of tokens in the prompt.
"""

import asyncio
import logging
import os
import sys
from io import BytesIO

from PIL import Image

from kani.engines.openai import OpenAIClient
from kani.ext.vision import ImagePart
from kani.ext.vision.engines.openai.models import OpenAIImage, OpenAIVisionChatMessage

sys.path.append("..")

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))


async def main():
    size = (512, 512)
    img = Image.new("RGB", size)
    img_bytes = BytesIO()
    img.save(img_bytes, "PNG")
    resp = await client.create_chat_completion(
        model="gpt-4-vision-preview",
        messages=[
            OpenAIVisionChatMessage(
                role="user",
                content=[
                    OpenAIImage.from_imagepart(ImagePart.from_bytes(img_bytes.getvalue())),
                ],
            )
        ],
        max_tokens=1,
    )
    print(f"{size}: {resp.prompt_tokens} tokens")
    print(resp)
    await client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
