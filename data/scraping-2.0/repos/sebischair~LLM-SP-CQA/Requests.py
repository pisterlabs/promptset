import openai
import time
import os
from typing import Tuple

def _send_to_chat(model: str, messages: list, local_server: bool = True, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    _set_openai_parameters(local_server)
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    response_content = response.choices[0].message.content
    end_time = time.time()

    duration = end_time - start_time
    return response_content, duration
    
def _set_openai_parameters(local_server: bool = True):
    if local_server:
        openai.api_key = "EMPTY"
        openai.api_base = "http://localhost:8000/v1"
    else:
        # Get OpenAI API key from .env file
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = "https://api.openai.com/v1"

def send_to_local_server_chat(messages: list, model: str, temperature: float = 0.0,  max_tokens: int = 64) -> Tuple[str, float]:
    ''' Sends the given prompt to the model that runs on the local server and returns the response as well as the execution time'''
    return _send_to_chat(model, messages, True, max_tokens, temperature)

def send_to_openai_chat(messages: list, model: str, max_tokens: int = 64, temperature: float = 0.0) -> Tuple[str, float]:
    ''' Sends the given messages to the OpenAI API chat endpoint and returns the response '''
    return _send_to_chat(model, messages, False, max_tokens, temperature)