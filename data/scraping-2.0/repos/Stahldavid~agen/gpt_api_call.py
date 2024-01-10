import time
import random
import openai

def api_call(messages, functions3, max_response_tokens):
    for i in range(15):
        try:
            return openai.ChatCompletion.create(
                model= "gpt-4",#"gpt-3.5-turbo-16k-0613",
                messages=messages,
                functions=functions3,
                temperature=0.7,
                max_tokens=max_response_tokens,
                function_call="auto"
            )
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            wait_time = 2 ** i + random.random()
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred: {e if isinstance(e, str) else repr(e)}")

            raise
    print("Maximum number of retries exceeded. Aborting...")
