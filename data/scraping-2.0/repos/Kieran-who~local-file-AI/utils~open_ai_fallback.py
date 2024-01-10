import openai
import asyncio
import time
from config import OPEN_AI_KEY, FALLBACK_OPEN_AI_MODEL

model_list = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k",
              "gpt-3.5-turbo-16k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-32k-0613", "gpt-4-32k", "gpt-4-0613", "gpt-4"]


async def get_vector_openai(text):
    for i in range(10):
        try:
            call_args = {
                "api_key": OPEN_AI_KEY,
                "api_type": "open_ai",
                "api_base": "https://api.openai.com/v1",
                "input": text,
                "model": "text-embedding-ada-002"
            }

            response = openai.Embedding.create(**call_args)
            embeddings = response['data'][0]['embedding']
            return embeddings
        except:
            # Increase exponential backoff delay between each retry attempt
            await asyncio.sleep((2 ** i) + (i ** 2))
            continue
    # If the code reaches here, all attempts have failed
    raise


async def get_open_ai_chat_completion(messages, model, max_res_tokens, temp, functions=None, function_to_call="auto"):
    if model not in model_list:
        model = FALLBACK_OPEN_AI_MODEL

    for i in range(10):
        try:
            # Initialize the arguments dictionary
            call_args = {
                "api_key": OPEN_AI_KEY,
                "api_type": "open_ai",
                "api_base": "https://api.openai.com/v1",
                "model": model,
                "messages": messages,
                "max_tokens": max_res_tokens,
                "temperature": temp,
            }

            # If functions parameter is passed, include "functions" and "function_call" in the arguments dictionary
            if functions is not None:
                call_args["functions"] = functions
                call_args["function_call"] = function_to_call

            response = openai.ChatCompletion.create(**call_args)
            return response['choices'][0]
        except:
            # Increase exponential backoff delay between each retry attempt
            await asyncio.sleep((2 ** i) + i ** 2)
            continue
    # If the code reaches here, all attempts have failed
    raise
