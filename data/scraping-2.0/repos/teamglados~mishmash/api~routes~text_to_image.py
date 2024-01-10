from api.app import *

import openai
import asyncio
import concurrent
import functools
import os

@functools.lru_cache
def dalle_create_image_with_auth(api_key, **kwargs):
    openai.api_key = api_key
    return openai.Image.create(
        response_format='b64_json',
        **kwargs
    )

async def dalle_create_image(*args, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exc:
        return await asyncio.get_running_loop().run_in_executor(
            exc,
            functools.partial(
                dalle_create_image_with_auth,
                os.environ['OPENAI_KEY'],
                **kwargs
            )
        )

@app.get('/text2image')
async def create(prompt: str = '', count: int = 1, size: int = 1):
    logger.info(f'text to image using the following prompt: {repr(prompt)}')
    size = [
        '256x256',
        '512x512',
        '1024x1024'
    ][min(max(0, size-1), 2)]
    result = await dalle_create_image(
        prompt=prompt,
        size=size,
        n=count,
    )
    return {'result': list(map(lambda r: r['b64_json'], result['data']))}
