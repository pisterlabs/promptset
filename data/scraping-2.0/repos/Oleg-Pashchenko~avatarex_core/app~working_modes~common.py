import dataclasses
import os

from openai import AsyncOpenAI


from abc import ABC, abstractmethod
import gdown

from app.utils import err
from app.utils.db import MethodResponse, Message, Command





class Mode(ABC):
    method_response = MethodResponse(data=[], all_is_ok=True, errors=set())

    @abstractmethod
    async def execute(self) -> MethodResponse:
        pass

    async def perephrase(self, message: str, openai_api_key: str) -> str:
        try:
            aclient = AsyncOpenAI(api_key=openai_api_key)
            response = await aclient.chat.completions.create(model='gpt-3.5-turbo',
            messages=[{"role": "system", "content": 'Перефразируй'},
                      {'role': 'user', 'content': message}],
            max_tokens=4000,
            temperature=1)
            return response['choices'][0]['message']['content']
        except:
            self.method_response.errors.add(err.OPENAI_REQUEST_ERROR)
            self.method_response.all_is_ok = False
            return message

    @staticmethod
    async def is_message_first(messages_history: list) -> bool:
        return len(messages_history) == 0

    @staticmethod
    def make_responses_unique(responses: list[Message | Command]):
        result, cache = [], []
        for response in responses:
            if isinstance(response, Message) and response.text not in cache:
                cache.append(response.text)
                result.append(response)
            else:
                result.append(response)
        return result

    async def download_file(self, file_link: str) -> bool:
        file_id = file_link.split("id=")[1]
        output_path = f"files/{file_id}.xlsx"

        if not os.path.exists(output_path):
            try:
                gdown.download(file_link, output_path, quiet=True)
                return True
            except:
                self.method_response.errors.add(err.UNABLE_TO_DOWNLOAD_FILE)
                self.method_response.all_is_ok = False
                return False

    async def qualification_passed(self):
        pass
