
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_DALLE_API_KEY :str
    AOAI_DALLE_ENDPOINT :str

settings = Settings(_env_file=".env")

from logger import logger

from openai import AsyncAzureOpenAI

async def call_dalle (prompt):

    client = AsyncAzureOpenAI(
        azure_endpoint = settings.AOAI_DALLE_ENDPOINT, 
        api_key=settings.AOAI_DALLE_API_KEY,  
        api_version="2023-12-01-preview",
        timeout=30,
        max_retries=1,
    )

    image = await client.images.generate(
        model="dalle3",
        prompt=prompt,
        n=1, #制限
    )

    logger.debug(image)

    image_url = image.data[0].url
#    image_url_2 = image.data[1].url
#    image_url_3 = image.data[2].url

    return image_url
#    return image_url_1, image_url_2, image_url_3

