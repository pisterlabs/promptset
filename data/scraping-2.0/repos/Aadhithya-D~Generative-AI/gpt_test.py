# Import the os package
import os

# Import the openai package
import openai

openai.api_key = "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"
response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": 'You are a AI engineer.'},
                        {"role": "user", "content": 'I want you to create a workflow for generating a chatbot, that gives fashion advice.'},
              ])


print(response["choices"][0]["message"]["content"])