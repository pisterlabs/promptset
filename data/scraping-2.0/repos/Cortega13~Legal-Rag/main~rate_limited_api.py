import openai
import random
import time

openai.api_key = ""

class RateLimitedAPI:
    def __init__(self, max_retries=3, base_delay=1.0, max_delay=60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def request(self, task):
        for i in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0613",
                    messages=task,
                    temperature=0.05,
                    max_tokens=1000,
                    frequency_penalty=0.1,
                )
                assistant_reply = response.choices[0].message["content"]
                return assistant_reply
            except Exception as e:
                delay = min(
                    self.max_delay,
                    (self.base_delay * 2**i) + (random.randint(0, 1000) / 1000.0),
                )
                print(f"Request failed with {str(e)}, retrying in {delay} seconds.")
                time.sleep(delay)
        else:
            print("Exhausted maximum retries")