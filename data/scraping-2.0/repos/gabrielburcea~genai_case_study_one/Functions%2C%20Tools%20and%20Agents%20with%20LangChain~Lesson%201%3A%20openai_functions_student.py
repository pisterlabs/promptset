# Databricks notebook source
# OpenAI Function Calling


# COMMAND ----------

# MAGIC %md
# MAGIC **Notes**:
# MAGIC - LLM's don't always produce the same results. The results you see in this notebook may differ from the results you see in the video.
# MAGIC - Notebooks results are temporary. Download the notebooks to your local machine if you wish to save your results.

# COMMAND ----------

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]

# COMMAND ----------

import openai

# COMMAND ----------

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions
)

# COMMAND ----------

print(response)

# COMMAND ----------

response_message = response["choices"][0]["message"]

# COMMAND ----------

response_message

# COMMAND ----------

response_message["content"]

# COMMAND ----------

response_message["function_call"]

# COMMAND ----------

json.loads(response_message["function_call"]["arguments"])

# COMMAND ----------

args = json.loads(response_message["function_call"]["arguments"])

# COMMAND ----------

get_current_weather(args)

# COMMAND ----------

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]

# COMMAND ----------

messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]

# COMMAND ----------

print(response)

# COMMAND ----------

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
    function_call="auto",
)
print(response)

# COMMAND ----------

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
    function_call="none",
)
print(response)

# COMMAND ----------


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
    function_call="none",
)
print(response)

# COMMAND ----------

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
    function_call={"name": "get_current_weather"},
)
print(response)

# COMMAND ----------

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

# COMMAND ----------

messages.append(response["choices"][0]["message"])

# COMMAND ----------

args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = get_current_weather(args)

# COMMAND ----------

messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)

# COMMAND ----------

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
)
print(response)

# COMMAND ----------


