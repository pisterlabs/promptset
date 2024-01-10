from openai.openai_object import OpenAIObject
from openai import ChatCompletion
import discord

import functools
import asyncio
import random

def random_chance(percent: int) -> bool:
    return random.randint(1, 100) <= percent

def filter_markdown(text: str) -> str:
    filtered = text.replace("\\", "\\\\") # escape backslashes first

    filtered = filtered.replace("*", "\\*")
    filtered = filtered.replace("_", "\\_")
    filtered = filtered.replace("~", "\\~")

    filtered = filtered.replace("`", "\\`")

    filtered = filtered.replace(">", "\\>")
    filtered = filtered.replace("|", "\\|")

    return filtered

def is_mentioned(user: discord.User, message: discord.Message) -> bool:
    if isinstance(message.channel, discord.DMChannel):
        return True

    if user.mentioned_in(message):
        return True

    return False

async def chat_request(messages: list, model: str = "gpt-3.5-turbo", loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()) -> OpenAIObject:
    function = functools.partial(ChatCompletion.create, model=model, messages=messages)
    
    response = await loop.run_in_executor(None, function)
    response = response.choices[0].message

    return response