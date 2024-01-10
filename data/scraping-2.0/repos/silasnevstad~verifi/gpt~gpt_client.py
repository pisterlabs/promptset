import openai
from decouple import config

class GPTClient:
    def __init__(self):
        self.api_key = config('OPENAI_API_KEY')
        openai.api_key = self.api_key
        self.RETRY_LIMIT = 3

    def chat_completion(self, messages, functions):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        return response

    def get_function_call(self, messages, functions):
        for attempt in range(self.RETRY_LIMIT):
            response = self.chat_completion(messages, functions)
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                message = choice.get('message', {})
                function_call = message.get('function_call')
                if function_call:
                    return function_call
            print(f"Retry {attempt + 1}/{self.RETRY_LIMIT}: No function call found.")
        print(f"Exceeded retry limit of {self.RETRY_LIMIT}. No function call found.")
        return None