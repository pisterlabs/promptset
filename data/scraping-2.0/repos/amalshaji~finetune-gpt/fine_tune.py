import asyncio
import timeit
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")


prompt = """
Generate a message in less than 50 words using the following parameters:
    occasion: $occasion
    tone: $tone

Use the following template tags as placeholders wherever necessary
| tags | description |
|---|---|
| {{ first_name }} | User's first name |
| {{ last_name }} | User's last name |
| {{ email }} | User's email |
"""


async def run(occasion: str, tone: str):
    start = timeit.default_timer()
    response = await openai.ChatCompletion.acreate(
        model="ft:gpt-3.5-turbo-0613:personal::819PHd1U",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"occasion: {occasion}, tone: {tone}",
            },
        ],
    )
    print(response)
    print(f"Time taken: {timeit.default_timer() - start}s")


asyncio.run(run(occasion="birthday", tone="informal"))
