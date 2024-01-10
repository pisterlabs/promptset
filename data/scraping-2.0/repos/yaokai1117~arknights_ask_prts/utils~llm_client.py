import openai
import os
ROOT_PATH = os.path.join(os.path.dirname(__file__), '..')

import sys
sys.path.append(ROOT_PATH)

from data_model import Message
from abc import ABC, abstractmethod
from typing import List
from dotenv import load_dotenv

class LlmClient(ABC):
    @abstractmethod
    def send(self, messags: List[Message]) -> str:
        pass

    @abstractmethod
    def model(self) -> str:
        pass

class Gpt35(LlmClient):
    MODEL = "gpt-3.5-turbo-16k"

    def __init__(self) -> None:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    # overriding abstract method
    def send(self, messages: List[Message]) -> str:
        response = openai.ChatCompletion.create(
            model=Gpt35.MODEL,
            messages=[message.model_dump() for message in messages],
            temperature=0,
        )
        choice = response['choices'][0]
        return choice['message']['content']

    # overriding abstract method
    def model(self) -> str:
        return Gpt35.MODEL

llm_client = Gpt35()
