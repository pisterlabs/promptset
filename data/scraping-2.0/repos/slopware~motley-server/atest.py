import asyncio
import openai
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def generate():
    stream = await client.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": "You are an AI Jerry Seinfeld."
        },
        {"role": "user", "content": "Tell me a joke."}
        ],
        stream=True,
    )
    async for chunk in stream:
        #print(chunk.choices[0].delta.content or "", end="")
        yield chunk.choices[0].delta.content or ""

async def main():
    async for text in generate():
        print(text, end="")

asyncio.run(main())