
import os
import typing
import logging

from dotenv import load_dotenv, find_dotenv

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# pip install -U wikipedia

from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__file__)

_ = load_dotenv(find_dotenv())


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0613"

import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)

# define a function
functions = [
    {
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
    }
]

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

response = openai.ChatCompletion.create(
    model=llm_model,
    messages=messages,
    functions=functions
)

print(response)

""" Output from above print
    {
  "id": "chatcmpl-8YuuRRUv8miTSJ8QFnDSiNoM7pMXE",
  "object": "chat.completion",
  "created": 1703332603,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\n\"location\": \"Boston, MA\"\n}"
        }
      },
      "logprobs": null,
      "finish_reason": "function_call"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 17,
    "total_tokens": 99
  },
  "system_fingerprint": null
}
"""

response_message = response["choices"][0]["message"]

# pull out the underline argument the llm asks in the message
args = json.loads(response_message["function_call"]["arguments"])

get_current_weather(args)

"""
'{"location": {"location": "Boston, MA"}, "temperature": "72", "unit": "fahrenheit", "forecast": ["sunny", "windy"]}'
"""

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
)

print(response)

"""
    {
  "id": "chatcmpl-8Yuwt5pAhRwdwVZ1fQ1X0EERxZ4o7",
  "object": "chat.completion",
  "created": 1703332755,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 76,
    "completion_tokens": 10,
    "total_tokens": 86
  },
  "system_fingerprint": null
}
"""

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto", # default value for this argument which lets the LLM decide if it should use function call
)
print(response)


"""
    {
  "id": "chatcmpl-8YuzC7VLkV39DINIGUPvM66N6FdnV",
  "object": "chat.completion",
  "created": 1703332898,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 76,
    "completion_tokens": 10,
    "total_tokens": 86
  },
  "system_fingerprint": null
"""


messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="none", yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
)
print(response)

"""
{
"id": "chatcmpl-8Yv03i9THyIptsC8v8344dRpPxr0h",
"object": "chat.completion",
"created": 1703332951,
"model": "gpt-3.5-turbo-0613",
"choices": [
{
    "index": 0,
    "message": {
    "role": "assistant",
    "content": "Hello! How can I assist you today?"
    },
    "logprobs": null,
    "finish_reason": "stop"
}
],
"usage": {
"prompt_tokens": 77,
"completion_tokens": 9,
"total_tokens": 86
},
"system_fingerprint": null
}
"""

messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="none", # forcing it not to try to call function via function call even though it should
)
print(response)

"""
{
  "id": "chatcmpl-8Yv1NdFro3zxlZhn7AUf4HYqpFG4y",
  "object": "chat.completion",
  "created": 1703333033,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Please give me a moment to check the weather in Boston."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 12,
    "total_tokens": 94
  },
  "system_fingerprint": null
}
"""

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"}, # forcing the llm to call a function even though it might not need to, causing it to hallucinate
)
print(response)

"""
{
  "id": "chatcmpl-8Yv1OMrrww2vlWMb655BuE0wLUrqy",
  "object": "chat.completion",
  "created": 1703333034,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\n  \"location\": \"San Francisco, CA\"\n}"
        }
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 83,
    "completion_tokens": 12,
    "total_tokens": 95
  },
  "system_fingerprint": null
}
"""


messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)

"""
{
  "id": "chatcmpl-8Yv1PzJaB4CmiikYb24JyczwqJE2e",
  "object": "chat.completion",
  "created": 1703333035,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "function_call": {
          "name": "get_current_weather",
          "arguments": "{\n  \"location\": \"Boston, MA\"\n}"
        }
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 89,
    "completion_tokens": 11,
    "total_tokens": 100
  },
  "system_fingerprint": null
}
"""

messages.append(response["choices"][0]["message"])

args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = get_current_weather(args)

# we call the underlying function ourselves and ensure to add a new message into the list of messages with a role of "function" 
# to indicate this is the response to a specific function call via the function name
messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
print(response)


"""
{
  "id": "chatcmpl-8Yv1SntkhF6iiB4MfzaZ5TyoiYial",
  "object": "chat.completion",
  "created": 1703333038,
  "model": "gpt-3.5-turbo-0613",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The weather in Boston, MA is currently sunny and windy with a temperature of 72\u00b0F."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 77,
    "completion_tokens": 19,
    "total_tokens": 96
  },
  "system_fingerprint": null
}
"""