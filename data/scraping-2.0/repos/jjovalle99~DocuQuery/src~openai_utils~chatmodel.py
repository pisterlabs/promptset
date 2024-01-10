import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-3.5-turbo-1106"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

    async def run(self, messages, text_only: bool = True):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        client = AsyncOpenAI()
        stream = await client.chat.completions.create(
            messages=messages, model=self.model_name, stream=True
        )

        return stream
