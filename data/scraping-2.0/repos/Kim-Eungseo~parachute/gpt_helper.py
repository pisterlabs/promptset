import os
from typing import List

import openai
import tiktoken
from dotenv import load_dotenv
from prompts import *

# Load the .env file
load_dotenv()

OpenAI_API_Key: str = os.environ.get("OPENAI_API_KEY")
openai.api_key = OpenAI_API_Key

MODEL_NAME = "gpt-3.5-turbo"


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(MODEL_NAME)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def __build_input_prompt_messages(text: str, system_prompt: str = None) -> List:
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": text})
    return messages


def send_to_gpt_chat_completion(message: str, system_prompt: str = None, temperature=1) -> str:
    messages = __build_input_prompt_messages(text=message, system_prompt=system_prompt)

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature
    )
    response = response['choices'][0]['message']['content']
    return response


def send_to_helper_agent(message: str) -> str:
    return send_to_gpt_chat_completion(message=message, system_prompt=HELPER_SYSTEM_PROMPT)
