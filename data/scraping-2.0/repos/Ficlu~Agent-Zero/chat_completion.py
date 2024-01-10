import os
import time
import openai

class ChatCompletion:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    # Expects 
    #   messages as list of dictionaries
    #   openai model that uses chat-endpoint
    # Outputs 
    #   string
    def complete(self, messages, model, temp=1.012, max_retries=4):
        for i in range(max_retries): 
            try:
                completion = openai.ChatCompletion.create(
                model=model,
                temperature=temp,
                messages=messages
                )
                return completion.choices[0].message.content
            except openai.error.RateLimitError: 
                print("Rate limit exceeded, retrying...")
                time.sleep((2 ** i) * 0.1)  # exponential backoff
            except openai.error.APIError:
                print("API error, retrying...")
                time.sleep((2 ** i) * 0.1)  # exponential backoff
        print(f"Failed after {max_retries} attempts")
        return None
