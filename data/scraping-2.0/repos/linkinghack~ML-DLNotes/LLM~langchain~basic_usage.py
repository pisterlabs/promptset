#! pip install python-dotenv openai
#! pip install --upgrade langchain

import os
import openai
import asyncio

from dotenv import load_dotenv, find_dotenv

# from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
llm_model = "gpt-3.5-turbo"

async def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        temperature=0.5
    )
    return await response


async def main():
    print(await get_completion("多元微分方程有哪些解法？"))

asyncio.run(main())