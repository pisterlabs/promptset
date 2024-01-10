from openai.types.chat.chat_completion import Choice
from typing import Literal

import logging
import dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

dotenv.load_dotenv()
logging.basicConfig(level=logging.DEBUG)


class _Completions:
    def __init__(self, real_client: AsyncOpenAI):
        self.real_client = real_client
        self.next_responses: list[ChatCompletion] = []

    async def create(self, *args, **kwargs) -> ChatCompletion:
        res = self.next_responses.pop(0)
        if res == "CALL_OPENAI":
            res = await self.real_client.chat.completions.create(*args, **kwargs)
        return res


class _Chat:
    def __init__(self, real_client: AsyncOpenAI):
        self.completions = _Completions(real_client=real_client)


class MockOpenAI:
    def __init__(self, real_client: AsyncOpenAI):
        self.chat = _Chat(real_client=real_client)

    def add_next_responses(self, *responses: ChatCompletionMessage | dict | Literal["CALL_OPENAI"]):
        msgs = [
            ChatCompletionMessage(**response) if isinstance(response, dict) else response
            for response in responses
        ]
        completions = [
            ChatCompletion(
                id="1",
                choices=[Choice(
                    index=0,
                    finish_reason="stop",
                    message=msg
                )],
                created=1,
                model="gpt-0",
                object="chat.completion"
            )
            for msg in msgs
        ]
        self.chat.completions.next_responses += completions