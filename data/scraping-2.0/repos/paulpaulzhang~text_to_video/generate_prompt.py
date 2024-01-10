import requests
import json
import openai
import time
from openai.error import (
    APIError,
    APIConnectionError,
    RateLimitError,
    Timeout,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError,
)


class PromptGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self, text):
        messages = self.config.llm_prompt.copy()
        messages.append({"role": "user", "content": text})
        try:
            openai.api_key = self.config.get_openai_key()
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )
            json_str = json.loads(result["choices"][0].message.content)
            time.sleep(5)
            return json_str
        except (
            APIError,
            APIConnectionError,
            RateLimitError,
            Timeout,
            ServiceUnavailableError,
            AuthenticationError,
            InvalidRequestError,
            json.decoder.JSONDecodeError,
        ) as e:
            print(e)
            return None


if __name__ == "__main__":
    from config import Config

    # with open("test.txt", "r", encoding="utf8") as f:
    #     lines = f.readlines()
    #     lines = "".join(lines).replace("\n\n", "\n").strip(" ").strip("\n").split("\n")
    #     lines = str(lines)
    # print(lines)
    generator = PromptGenerator(Config())
    print(generator.generate("江城步入教室，和兄弟们一起找了个角落坐下。"))
