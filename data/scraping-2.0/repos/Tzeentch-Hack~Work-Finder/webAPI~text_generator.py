import os
import time

import openai
from dotenv import load_dotenv

load_dotenv()

#openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION_KEY")


class TextGenerator:
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens

    def get_response(self, system: str, prompt: str) -> str:
        while True:
            try:
                completions = openai.ChatCompletion.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completions.choices[0].message.content
            except (openai.error.RateLimitError, openai.error.APIConnectionError):
                print(
                    "Rate limit reached or connection error occurred. Waiting one second before retrying."
                )
                time.sleep(1)
