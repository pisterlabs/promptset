import os
from dotenv import load_dotenv
import openai
import requests
import asyncio
import json
import time
import asyncio
import httpx

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def picGen(dalle: bool, prompt: str):
    if dalle:
        return await call_dalle_api(prompt)
    else:
        return await unsplash_it(prompt)


async def call_dalle_api(prompt):
    print("running")
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "url",
            },
        )

        response = response.json()

        print("its working x")
        return response["data"][0]["url"]


# async def unsplash_it(query):
#     url = f"https://edcomposer.vercel.app/api/getGoogleResult?search={query}"
#     response = await requests.get(url)
#     return response.json()[0]


async def unsplash_it(query):
    url = f"https://edcomposer.vercel.app/api/getGoogleResult?search={query}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()[0]


test_array = [
    {"prompt": "Phases of the Revolution", "generate": True},
    {
        "prompt": "The Estates-General convened in 1789, leading to the formation of the National Assembly, representing the common people's interests.",
        "generate": False,
    },
    {
        "prompt": "The National Constituent Assembly (1789-1791) drafted the Constitution of 1791, establishing a constitutional monarchy.",
        "generate": True,
    },
]

return_array = []


async def process_data(inp_array):
    start = time.time()

    tasks = [
        picGen(currObj.get("generate"), currObj.get("prompt")) for currObj in inp_array
    ]
    return_array = await asyncio.gather(*tasks)
    end = time.time()
    print(end - start)
    return return_array


def generatePictures(inp_array):
    for currObj in inp_array:
        print(currObj.get("prompt"))
    loop = asyncio.get_event_loop()
    return_array = loop.run_until_complete(process_data(test_array))

    return return_array


print(generatePictures(test_array))