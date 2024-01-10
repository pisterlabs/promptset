"""This module is deprecated and will be removed in a future release."""
# add the project root directory to the system path
if __name__ == "__main__":
    from pathlib import Path

    project_directory = Path(__file__).parent.parent
    import sys

    # sys.path.insert(0, str(project_directory))
    sys.path.append(str(project_directory))

import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

from controller.create_logger import create_logger
from controller.load_openai import load_openai

# Create a logger for this module
module_logger = create_logger(
    logger_name="controller.utilities",
    logger_filename="utilities.log",
    log_directory="logs",
    add_date_to_filename=False,
)

# Load the OpenAI API key from the .env file
load_openai()

GPT_MODEL = "gpt-3.5-turbo-0613"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, functions=None, function_call=None, model=GPT_MODEL
):
    """Send a request to the OpenAI API to generate a chat response."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
            timeout=10,
        )
        module_logger.debug("response: %s\n%s", response, response.json())
        return response
    except Exception as error:
        module_logger.debug("Unable to generate ChatCompletion response")
        module_logger.error("Exception: %s", error)
        return error


def pretty_print_conversation(messages):
    """Print a conversation in a human-readable format."""
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "user":
            print(
                colored(f"user: {message['content']}\n", role_to_color[message["role"]])
            )
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['function_call']}\n",
                    role_to_color[message["role"]],
                )
            )
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(
                colored(
                    f"assistant: {message['content']}\n", role_to_color[message["role"]]
                )
            )
        elif message["role"] == "function":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )


def main():
    """Run the main function."""
    # while loop to use input to send messages to the chatbot
    messages = []
    while True:
        message = input("User: ")
        # if the user types "quit", exit the while loop
        if message == "quit":
            break
        messages.append({"role": "user", "content": message})
        response = chat_completion_request(messages=messages)
        response_json = response.json()
        try:
            messages.append(
                {"role": "system", "content": response_json["choices"][0]["message"]["content"]}
            )
        except Exception as error:
            messages.append({"role": "system", "content": error})
            pretty_print_conversation(messages=messages)
        else:
            pretty_print_conversation(messages=messages)
    print("Goodbye!")


if __name__ == "__main__":
    main()

    res = {
        "id": "chatcmpl-7jzcIBYE3LBhWTJfGqB1J2Z47us2i",
        "object": "chat.completion",
        "created": 1691195970,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "¡Hola! ¿En qué puedo ayudarte hoy?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20},
    }
    message = res["choices"][0]["message"]["content"]