import aiohttp
import base64
import os
import json  # Make sure to import json
from openai import AsyncOpenAI
import logging

client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

async def dalle_image_request(prompt, quality, style):
    try:
        # Await the response from the async call
        response = await client.images.generate(
            prompt=prompt,
            quality=quality,
            style=style,
            response_format='b64_json',
            model='dall-e-3',
            size='1024x1024',
            n=1,
        )

        # Convert the response to JSON
        data_str = response.json()
        data = json.loads(data_str)

        # Decode the base64 image and get the revised prompt
        image_data = base64.b64decode(data['data'][0]["b64_json"])
        revised_prompt = data['data'][0]['revised_prompt']

        return image_data, revised_prompt

    except Exception as e:
        print(f"Error in generating image: {e}")
        return None, None