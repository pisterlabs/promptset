from typing import Generator
import openai

from hai.mock import dummyai


class ApiManager:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def get_ai_response(
        self, conversation: list[str]
    ) -> Generator[str, None, None]:
        response = None
        if self.model_name == "mock":
            response = dummyai.DummyAI.create(
                conversation=conversation, stream=True
            )
            for chunk in response:
                try:
                    yield chunk
                except:
                    return
        else:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=conversation,
                stream=True,
            )
            for chunk in response:
                try:
                    yield chunk["choices"][0]["delta"]["content"]
                except:
                    return
