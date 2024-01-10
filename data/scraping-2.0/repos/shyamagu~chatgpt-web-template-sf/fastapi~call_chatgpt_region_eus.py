
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_EUS_API_KEY :str
    AOAI_EUS_ENDPOINT :str
    AOAI_EUS_MODEL :str

settings = Settings(_env_file=".env")

from logger import logger

import aiohttp

async def call_chatgpt_eus_raw (messages,temperature=0):
    try:
        url = f"{settings.AOAI_EUS_ENDPOINT}openai/deployments/{settings.AOAI_EUS_MODEL}/chat/completions?api-version=2023-05-15"
        headers = {
        "Content-Type": "application/json",
        "api-key": settings.AOAI_EUS_API_KEY
        }
        data = {
        "messages": messages,
        "temperature": temperature,
        }

        async with aiohttp.ClientSession() as session:
            response = await session.post(url, headers=headers, json=data)
            response = await response.json()
    except Exception as e:
        logger.debug(e)
        
    return response['choices'][0]['message']['content']

from openai import AsyncAzureOpenAI

async def call_chatgpt_eus (messages,temperature=0):

    client = AsyncAzureOpenAI(
        azure_endpoint = settings.AOAI_EUS_ENDPOINT, 
        api_key=settings.AOAI_EUS_API_KEY,  
        api_version="2023-05-15",
        timeout=20,
    )

    response = await client.chat.completions.create(
        model=settings.AOAI_EUS_MODEL,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content