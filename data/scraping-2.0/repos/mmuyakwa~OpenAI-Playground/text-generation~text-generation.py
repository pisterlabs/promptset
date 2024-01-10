# Description: This file contains the code for generating text using the OpenAI API
# Source: https://platform.openai.com/docs/guides/text-generation

import json
from openai import OpenAI
client = OpenAI()

# Create an instance of the OpenAI client
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)


# The output of "response" is an object of the type "CompletionCreateResponse", which contains the following attributes:
"""
ChatCompletion(id='chatcmpl-8KsXlRSIKVauwoXdnAr9JTFEOSAKX', choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='The World Series in 2020 was played at Globe Life Field in Arlington, Texas.', role='assistant', function_call=None, tool_calls=None))], created=1699986917, model='gpt-3.5-turbo-0613', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=53, total_tokens=71))
"""

# If finish_reason != "content_filter"
if response.choices[0].finish_reason != "content_filter":
    # Print the generated text "content" from the response
    print(response.choices[0].message.content)
else:
    print("The text is deemed offensive by OpenAI.")

