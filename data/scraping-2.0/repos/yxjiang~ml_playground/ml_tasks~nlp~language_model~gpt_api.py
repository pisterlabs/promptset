from abc import ABC, abstractmethod
import os
import openai
import requests
from typing import Any


class OpenAICommunicator(ABC):
    def __init__(self):
        localtion = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.secret_key = None
        with open(os.path.join(localtion, "openai.secrets")) as f:
            self.secret_key = f.readline()
        if not self.secret_key:
            raise ValueError(f'Secrete is null.')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.secret_key}',
        }

    def list_model(self):
        return openai.Model.list()

    @abstractmethod
    def send_requests(self, prompt: Any):
        pass


class OpenAICompleteCommunicator(OpenAICommunicator):

    def __init__(self, max_token: int = 100, temperature: float = 0.5, model_type: str = 'text-ada-001'):
        super().__init__()
        self.max_token = max_token
        self.temperature = temperature
        available_models = [
            'text-ada-001',      # $0.0004 / 1K tokens
            'text-babbage-001',  # $0.0005 / 1K tokens
            'text-curie-001',    # $0.002 / 1K tokens
            'text-davinci-003'   # $0.02 / 1K tokens
        ]
        if model_type not in available_models:
            raise ValueError(f'Only accept models in {available_models}')
        self.model_type = model_type

    
    def send_requests(self, prompt: str) -> str:
        data = {
            'model': self.model_type,
            'prompt': prompt,
            'max_tokens': self.max_token,
        }
        response = requests.post(url='https://api.openai.com/v1/completions', headers=self.headers, json=data)
        if response.status_code != 200:
            print(response.json())
            return "Error"
        json = response.json()
        answer = json['choices'][0]['text']
        print(answer)
        return answer
        

class OpenAIChatCommunicator(OpenAICommunicator):
    def __init__(self, max_token: int = 30, temperature: float = 0.5, model_type: str = 'gpt-3.5-turbo'):
        super().__init__()
        self.max_token = max_token
        self.temperature = temperature
        available_models = [
            'gpt-3.5-turbo',     # $0.002 / 1K tokens
            'gpt-3.5-turbo-0301' # $0.002 / 1K tokens
        ]
        if model_type not in available_models:
            raise ValueError(f'Only accept models in {available_models}')
        self.model_type = model_type
        self.previous_question = None
        self.previous_answer = None

    def send_requests(self, prompt: str) -> str:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
        ]
        if self.previous_question:
            messages.append({'role': 'user', 'content': self.previous_question})
            messages.append({'role': 'assistant', 'content': self.previous_answer})
        messages.append({'role': 'user', 'content': prompt})

        data = {
            'model': self.model_type,
            'messages': messages,
        }
        response = requests.post(url='https://api.openai.com/v1/chat/completions', headers=self.headers, json=data)
        if response.status_code != 200:
            print(response.json())
            return "Error"
        json = response.json()
        answer = json['choices'][0]['message']['content']
        # Update the context.
        self.previous_question = prompt
        self.previous_answer = answer
        print(answer)
        return answer

if __name__ == "__main__":
    communicator = OpenAIChatCommunicator()
    # communicator.send_requests(prompt='What is the diameter of the earth and sun respectively?')
    communicator.send_requests(prompt='Who is the 43th president of the US?')
    communicator.send_requests(prompt='Who is before him?')