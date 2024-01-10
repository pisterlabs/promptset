import json
from prompts import *
from openai import OpenAI
import pprint
from openai import ChatCompletion


def system_message(message: str) -> dict[str, str]:
    return {"role": "system", "content": message}


def ai_message(response: ChatCompletion) -> dict[str, str]:
    return {"role": "assistant", "content": response.choices[0].message.content}


def user_message(message: str) -> dict[str, str]:
    return {"role": "user", "content": message}


def print_message_and_tokens(response: ChatCompletion):
    print(f"\n {response.choices[0].message.content}" f"\t => : {response.usage}")


# program setup

client = OpenAI()


ai_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

ai_tool_choice = "auto"

messages = [
    system_message(assistant_message),
]

# program start

response = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages)

print_message_and_tokens(response)
messages.append(ai_message(response))

user_input = input("Enter your message: ")
while user_input != "quit":
    messages.append(user_message(user_input))
    response = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages)
    print_message_and_tokens(response)
    messages.append(ai_message(response))
    user_input = input("Enter your message: ")
