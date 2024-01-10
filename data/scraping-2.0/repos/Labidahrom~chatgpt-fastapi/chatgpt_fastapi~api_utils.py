import asyncio
from chatgpt_fastapi.randomizer import RANDOMIZER_STRINGS
from dotenv import load_dotenv
from httpx import AsyncClient
import openai
import os
import random
import time

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API")
TEXTRU_KEY = os.getenv("TEXTRU_KEY")
TEXTRU_URL = os.getenv("TEXTRU_URL")


async def add_randomize_task(randomize_strings):
    return '\n'.join([random.choice(strings) for strings in randomize_strings])


async def make_async_textru_call(*args, **kwargs):
    async with AsyncClient() as client:
        result = await client.post(*args, **kwargs)
    return result


async def get_text_from_openai(task, temperature):
    await asyncio.sleep(5)
    local_time = time.localtime(time.time())
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    print(f'send_to_openai, time: {formatted_time}')
    randomize_string = await add_randomize_task(RANDOMIZER_STRINGS)
    full_task = task + '\n' + randomize_string
    client = openai.AsyncOpenAI(
        api_key=OPENAI_API_KEY,
    )
    try:
        text_request = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_task,
                }
            ],
            model="gpt-3.5-turbo",
            temperature=float(temperature)
        )
        text = str(text_request.choices[0].message.content)
        return {
            'text': text,
            'status': 'ok'
        }
    except openai.RateLimitError as e:
        error_details = f'Rate limit error: {e}'
    except openai.AuthenticationError as e:
        error_details = f'Authentication token error: {e}'
    except openai.APIConnectionError as e:
        error_details = f'Unable to connect to OpenAI server: {e}'
    except Exception as e:
        error_details = f'General OpenAI error: {e}'
    return {
        'text': None,
        'status': error_details
    }


async def add_text(text, text_len):
    task = (f'Допиши пожалуйста следующий текст,'
            f'что бы его длина составила {text_len}'
            f' символов или больше:\n{text}')
    return await get_text_from_openai(task, temperature=1)


async def raise_uniqueness(text, rewriting_task):
    task = f'{rewriting_task}\n{text}'
    return await get_text_from_openai(task, temperature=1)


async def get_text_uniqueness(text):
    await asyncio.sleep(10)
    local_time = time.localtime(time.time())
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    print(f'send_to_text.ru, time: {formatted_time}')
    headers = {
        "Content-Type": "application/json"
    }
    text_data = {
        "text": text,
        "userkey": TEXTRU_KEY
    }
    response = await make_async_textru_call(TEXTRU_URL,
                                            json=text_data,
                                            headers=headers)
    uid = response.json().get('text_uid')
    if not uid:
        return 0
    attempts = 0
    uid_data = {
        "uid": uid,
        "userkey": TEXTRU_KEY
    }
    while attempts < 10:
        # due to the slow work of text.ru api, we wait and periodically send
        # request to server until we get right response
        attempts += 1
        if attempts <= 5:
            # first five attempts have longer time intervals
            await asyncio.sleep(20)
        await asyncio.sleep(60)
        response = await make_async_textru_call(TEXTRU_URL,
                                                json=uid_data,
                                                headers=headers)
        if 'text_unique' in response.json():
            return float(response.json()["text_unique"])
    return 0
