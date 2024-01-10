import openai
import os
import time
import json
import asyncio
import os

async def send_to_chatGPT(message):

    # completion = await asyncio.get_event_loop().create_task(openai.ChatCompletion.acreate(
    #     model="gpt-3.5-turbo",
    #     messages=message
    # ))
    completion = await asyncio.get_event_loop().create_task(openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=message
    ))
    # await task
    # time.sleep(5)
    return completion.choices[0].message["content"]

async def create_image_from_GPT(message):
    completion = await openai.Image.acreate(
        prompt=message,
        n=1,
        size="512x512"
    )
    return completion.data[0]["url"]


async def edit_text(message, instruction):
    completion = await openai.Edit.acreate(
        model="text-davinci-edit-001",
        input=message,
        instruction=instruction
    )
    return completion.choices[0]["text"]
