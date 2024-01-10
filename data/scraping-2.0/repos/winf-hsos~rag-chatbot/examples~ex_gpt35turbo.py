import openai
# Add and load local chatbot module
import os
import sys
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_directory)

import chatbot as cb
openai.api_key = cb.OPENAI_API_KEY

messages=[
     {"role": "system", "content": "You are a helpful assistant and always kind."},
     {"role": "user", "content": "Who founded Microsoft?"}
     ]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
    )

print(response)


