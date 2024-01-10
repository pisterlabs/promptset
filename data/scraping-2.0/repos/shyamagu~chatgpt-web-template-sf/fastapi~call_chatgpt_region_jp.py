
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_JP_API_KEY :str
    AOAI_JP_ENDPOINT :str
    AOAI_JP_MODEL :str

settings = Settings(_env_file=".env")

from logger import logger

import aiohttp

async def call_chatgpt_jp_raw (messages,temperature=0):
    try:
        url = f"{settings.AOAI_JP_ENDPOINT}openai/deployments/{settings.AOAI_JP_MODEL}/chat/completions?api-version=2023-05-15"
        headers = {
        "Content-Type": "application/json",
        "api-key": settings.AOAI_JP_API_KEY
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

async def call_chatgpt_jp (messages,temperature=0):

    client = AsyncAzureOpenAI(
        azure_endpoint = settings.AOAI_JP_ENDPOINT, 
        api_key=settings.AOAI_JP_API_KEY,  
        api_version="2023-05-15",
        timeout=20,
    )

    response = await client.chat.completions.create(
        model=settings.AOAI_JP_MODEL,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content
