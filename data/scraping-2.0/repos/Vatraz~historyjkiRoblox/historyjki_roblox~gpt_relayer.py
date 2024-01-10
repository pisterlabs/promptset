import os
from typing import List

import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


class GtpRelayerException(Exception):
    pass


class GtpRelayer:
    def simply_ask(self, message: str) -> str:
        chat_messages = [self._create_message(message)]
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=chat_messages
            )
            answer = chat["choices"][0]["message"]["content"]
        except Exception as exe:
            raise GtpRelayerException(
                f"Failed to fetch ChatGTP response: {message}"
            ) from exe
        return answer

    def generate_image(
        self, prompt: str, n: int = 1, size: int = 256, response_format: str = "url"
    ) -> List[str]:
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=n,
                size=f"{size}x{size}",
                response_format=response_format,
            )
            image = response["data"][0][response_format]
        except Exception as ex:
            raise GtpRelayerException(
                f"Failed to fetch Image response: {prompt}"
            ) from ex
        return image

    def _create_message(self, message: str) -> dict:
        return {"role": "user", "content": message}
