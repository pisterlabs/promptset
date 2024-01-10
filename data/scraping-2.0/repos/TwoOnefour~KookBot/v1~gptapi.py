import openai
import os
import time
import json
import asyncio
import os

async def send_to_chatGPT(message):
    openai.proxy = {
        "http": "http://127.0.0.1:2599",
    }
    completion = await asyncio.get_event_loop().create_task(openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=message
    ))
    # await task
    # time.sleep(5)
    return completion.choices[0].message["content"]
