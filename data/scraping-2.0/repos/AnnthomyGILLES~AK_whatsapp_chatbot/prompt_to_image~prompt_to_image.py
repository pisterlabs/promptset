import asyncio
import logging

import aiohttp
import openai
from openai import AsyncOpenAI
from ratelimit import sleep_and_retry, limits

from utils import load_config

ONE_MINUTE = 60
MAX_CALLS_PER_MINUTE = 30
MAX_TOKENS = 400

load_config()

# Initialize the OpenAI client
client = AsyncOpenAI()


async def generate_image(prompt):
    async with aiohttp.ClientSession() as session:

        @sleep_and_retry
        @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
        async def request():
            try:
                print(prompt)
                response = await client.images.generate(
                    prompt=prompt,
                    size="256x256",
                    quality="standard",
                    n=1,
                )
                image_url = response.data[0].url

                return image_url
            except openai.RateLimitError as e:
                logging.error("Rate limit reached for DALL-E")

        return await request()


# Example usage
# asyncio.run(generate_image("a white siamese cat"))


async def main():
    PROMPT = "An eco-friendly computer from the 90s in the style of vaporwave"
    response = await generate_image(PROMPT)
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
