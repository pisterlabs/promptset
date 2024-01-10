
import logging
import openai

class chat_complete:

    def __init__(self, user_content, model="gpt-3.5-turbo-16k", max_tokens=15000, temperature=0.1, functions=None, system_content="You are a helpful assistant"):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.functions = functions if functions is not None else []
        self.messages = [{"role": "user", "content": user_content}]
        if system_content:
            self.messages.insert(0, {"role": "system", "content": system_content})

        self.completion = self.call_openai_create()
        
    
    def call_openai_create(self):
        task = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self.messages
        }

        # Only add 'functions' key if self.functions is not empty
        if self.functions:
            task['functions'] = self.functions
            task['function_call'] = {"name": self.functions[0]["name"]}

        # Here we use OpenAI's API to create a completion
        # completion = openai.ChatCompletion.create(**task)  # As per your original code
        completion = openai.ChatCompletion.create(**task)  # As per OpenAI's API as of September 2021
        logging.info (completion)
        return completion.to_dict()