import openai
import os
from typing import List

class GPT:
    def __init__(self, api_key: str, model: str = 'gpt-4', temperature: float = 1):
        openai.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.conversation = []

    def add_message(self, role: str, content: str):
        self.conversation.append({"role": role, "content": content})

    def get_reply(self, message: str):
        self.add_message("user", message)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.conversation,
                temperature=self.temperature
            )
            self.conversation = []  # Clear the conversation after each reply
            if not response.choices:
                return "Ошибка: OpenAI не вернул ожидаемый ответ."
            return response.choices[0].message['content']
        except openai.error.OpenAIError as e:
            return f"Ошибка OpenAI: {str(e)}"
        except Exception as e:
            return f"Общая ошибка: {str(e)}"

gpt_model = GPT(api_key='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')
