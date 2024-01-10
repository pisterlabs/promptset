import asyncio
import openai_async

from usagiBot.src.UsagiErrors import OpenAIError
from pycord18n.extension import _


class BaseAI:
    def __init__(self, api_key):
        self._api_key = api_key


class OpenAIHandler(BaseAI):

    def __init__(self, api_key):
        super().__init__(api_key)

        # Default values for gpt model
        self._ai_model = "gpt-3.5-turbo"

    async def get_ai_model(self):
        return self._ai_model

    async def generate_answer(self, question: str, counter: int = 0):
        try:
            response = await openai_async.chat_complete(
                self._api_key,
                timeout=200,
                payload={
                    "model": await self.get_ai_model(),
                    "messages": [{"role": "user", "content": question}],
                },
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

            retry_codes = [500, 429, 502]
            if response.status_code in retry_codes:
                if counter != 20:
                    await asyncio.sleep(2)
                    return await self.generate_answer(question, counter + 1)
                else:
                    return _("Something went wrong")
            else:
                raise OpenAIError(response.json().get('error'), response.status_code)

        except OpenAIError as error_answer:
            return error_answer

        except Exception as e:
            return str(e)
