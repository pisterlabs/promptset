# agents.py
from openai import OpenAI

class GeneratorAgent:
    def __init__(self, model:str, system_init_content:str, max_tokens: int):
        self.model = model
        self.system_init_content = system_init_content
        self.max_tokens = max_tokens
        self.system_prompt = {
            "role": "system",
            "content": system_init_content
        }

    def __call__(self, input_prompt:str):
        prompt = {
            "role": "user",
            "content": input_prompt
        }
        response = self.__chat_completion(prompt)
        return response

    def __chat_completion(self, prompt):
        client = OpenAI()
        messages = [self.system_prompt, prompt]
        completion = client.chat.completions.create(
            model = self.model,
            messages = messages,
            max_tokens = self.max_tokens
        )
        response = completion.choices[0].message.content
        return response

