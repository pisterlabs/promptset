from typing import List, Union
from tiktoken import get_encoding
from langchain.schema import BaseMessage

def count_tokens(messages: List[BaseMessage], model="gpt-3.5-turbo-0613") -> int:
    encoding = get_encoding("cl100k_base")
    tokens_per_message, tokens_per_name = 0, 0

    if model in ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return count_tokens(messages, "gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return count_tokens(messages, "gpt-4-0613")
    else:
        raise Exception(f"num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.__dict__.items():
            if value is not None:  # Check if value is not None
                str_value = str(value)  # Convert value to string
                num_tokens += len(encoding.encode(str_value))
                if key == "name":
                    num_tokens += tokens_per_name

    num_tokens += 3
    return num_tokens