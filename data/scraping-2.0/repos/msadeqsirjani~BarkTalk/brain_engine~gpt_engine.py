import os
import openai
import asyncio
from config import OPENAI_KEY

pre_content = """Note that if you cannot answer the question which is about something you do not know such as time-sensitive information(e.g. today's weather/stock, .etc), you can only reply \"IDK\" in your response without other characters. Do not say something like As an AI language model..., I'm sorry... and etc."""


class GptEngine:
    def __init__(self):
        openai.api_key = OPENAI_KEY
        self.messages = [
            {"role": "system", "content": pre_content},
        ]

    def command(self, prompt: str) -> str:
        prompt += "\n"

        self.messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )

        content = response.choices[0]['message']['content']
        self.messages.append({"role": "assistant", "content": content})

        return content.replace("\n", "")
