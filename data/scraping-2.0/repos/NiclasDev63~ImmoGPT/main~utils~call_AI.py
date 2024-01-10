import os
import openai
from utils import memory

openai.api_key = os.getenv("OPENAI_KEY")

def make_request(message_history: memory) -> str:
    """
    Makes a request to one Provder from provider_list

    Args:
        message_history: The message history between the user, system and assistant

    Returns:
        str: The answer to the latest prompt

    Raises:
        TypeError: If the message_history has the wrong format
    """
    
    if isinstance(message_history, list) and len(message_history) > 0 and \
        isinstance(message_history[0], dict):
        completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages= message_history
                    )

        return completion.choices[0].message.content
    raise TypeError("message_history has the wrong format")
