import openai

from typing import List
from language_model.base_model import BaseModel
from log_factory.logger import create_logger

from opentelemetry.trace import Tracer


logger = create_logger(__name__)

class ChatGPTModel (BaseModel):
    def __init__(self, prompt: str):
        self.system_prompt = { "role": "system", "content": prompt }
        self.history = []

    def set_prompt(self, prompt: str):
        self.system_prompt = { "role": "system", "content": prompt }

    def set_history(self, history: List):
        self.history = history
        self.history.reverse() # The history needs to be reversed in order to apply properly
    
    def complete(self, prompt: str) -> str:
        messages=[
            self.system_prompt,
            *self.history,
            { "role": "user", "content": prompt },
        ]
        completion =  openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        return completion.choices[0].message.content
    
        
if __name__ == "__main__":
    model = ChatGPTModel("You are ChatGPT, a large language model trained by OpenAI. You answer questions and when the user asks code questions, you will answer with code examples in markdown format.")
    completion = model.complete("How do I write a fastAPI program in python?")
    print(completion)