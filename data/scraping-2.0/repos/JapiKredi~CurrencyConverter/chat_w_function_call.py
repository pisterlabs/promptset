# Dependencies
# !pip install scipy
# !pip install tenacity
# !pip install tiktoken
# !pip install termcolor 
# !pip install openai
# !pip install requests

import openai
import os
import json
import openai
import os
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from dotenv import load_dotenv

# Read the OpenAI Api_key
openai.api_key = open("OPENAI_API_KEY.txt", "r").read().strip()

# This function visually displays a conversation by printing each message with its role and content in a specific color, making it easier to distinguish between different roles in the conversation.
# The pretty_print_conversation function takes a list of messages as input and prints each message in a formatted way, assigning different colors to different roles in the conversation.
# Defines a dictionary role_to_color that maps each role ("system", "user", "assistant", "tool") to a corresponding color ("red", "green", "blue", "magenta").
# Iterates over each message in the messages list. For example: If the role of the message is "system", it prints the message with the prefix "system:" in the corresponding color.
# If the role of the message is "assistant" and the message has a key "function_call", it prints the message with the prefix "assistant:" and the value of the "function_call" key in the corresponding color.

def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


# the @retry decorator is being used to retry the decorated function in case of failures or exceptions. 
# It provides a way to automatically retry the function multiple times with a delay between each retry.
# The wait parameter specifies the wait strategy to use between retries.
# Delay between retries will be a random exponential backoff with a multiplier of 1 and a maximum delay of 40 seconds.
# The function will be retried for a maximum of 3 attempts. 
# If the function still fails after 3 attempts, the exception will be raised.
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))


# The chat_completion_request function is a Python function that sends a chat completion request to the OpenAI API. 
# It takes several parameters:
# 1: messages: A list of messages representing the conversation history. Each message in the list is a dictionary with a role (either "system", "user", or "assistant") and content (the text of the message).
# 2: tools: A list of tools to use in the assistant. Each tool is a dictionary with a name and description.
# 3: tool_choice: The name of the tool to use in the assistant.
# 4: model: The name of the model to use for the chat completion request.
# The function returns a response object from the OpenAI API.
# The response object contains the following attributes:
# 1: id: The ID of the chat completion request.
# 2: object: The type of object returned by the API.
# 3: created: The date and time the chat completion request was created.
# 4: model: The name of the model used for the chat completion request.
# 5: choices: A list of choices returned by the chat completion request. Each choice is a dictionary with a text attribute containing the text of the message.
# 6: created: The date and time the chat completion request was created.
# 7: model: The name of the model used for the chat completion request.

# Hereby the steps taken by the function
# 1: Inside the function, it prepares the necessary headers for the API request, including the content type and authorization using the OpenAI API key.
# 2: It then creates a JSON payload (json_data) containing the model, messages, and optional tools and tool choice.
# The function makes a POST request to the OpenAI API endpoint https://api.openai.com/v1/chat/completions with the headers and JSON payload. 
# It expects to receive a response from the API.
# f the request is successful, the function returns the response. 
# If an exception occurs during the request, it prints an error message and returns the exception object.

def chat_completion_request(messages, tools=None, tool_choice=None, model='gpt-3.5-turbo-1106'):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_currency_symbol",
            "description": "Get the current currency symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_symbol": {
                        "type": "string",
                        "description": "The currency symbol, e.g. USD for US Dollar"
                    }
                },
                "required": ["currency_symbol"]
            }
        }
    }
]

# In the following code we are forcing the model to use the get_n_day_weather_forecast function by specifying the tool_choice parameter in the chat_completion_request function.
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What currency do you want to use?"})
messages.append({"role": "system", "content": "Please try to get the currency the user wants to use, and convert this in the currecny symbol that consists of 3 capital letters"})

chat_response = chat_completion_request(
    messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_currency_symbol"}}
)

assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
print(assistant_message)
pretty_print_conversation(messages)








