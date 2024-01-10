# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
import json


openai.api_key = "api-key"
openai.api_version = "2023-03-15-preview"
openai.api_type = "azure"
openai.api_base = "https://llmaugmenter.openai.azure.com/"

messages= [
    {"role": "user", "content": "Test input -- respond with Hello world if succesful!"}
]

response = openai.ChatCompletion.create(
    engine="gpt-35-turbo",
    messages=messages
)

print(response['choices'][0]['message'])