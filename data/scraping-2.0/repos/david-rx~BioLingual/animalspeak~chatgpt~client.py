import openai
from dataclasses import dataclass
from typing import List, Dict

import requests
import json
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class ChatHistory:
    messages: List[Dict]

    @classmethod
    def from_jsonl(cls, file_path):
        with open(file_path, "r") as file:
            chat_history = [json.loads(line) for line in file]
        return cls(messages=chat_history)

class ChatGPTClient:
    def __init__(self):
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def send_message(self, messages: List[Dict]):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        return response

    def chat(self, messages):
        response = self.send_message(messages)
        return response["choices"][0]["message"]["content"]

    
