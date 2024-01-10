import queue
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
import asyncio

# avoid ProactorEventLoop issues on windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

class GPTController:
    def __init__(self, api_key):
        self.client = AsyncOpenAI(api_key=api_key)
        self.queue = queue.Queue(maxsize=50)
    
    async def send_prompt(self, prompt):
        try:
            stream = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    self.queue.put_nowait(chunk.choices[0].delta.content)
        except:
            print('exception send prompt')
