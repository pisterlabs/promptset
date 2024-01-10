import openai
from fastapi import HTTPException, status

from src.conf.config import settings
from src.constants import *


openai.api_key = settings.openai_api_key


async def generate_image(description: str):
    try:
        url = openai.Image.create(
            prompt=description,
            n=1,
            size="1024x1024"
            )['data'][0]['url']
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NO_TOKENS)
    return url
