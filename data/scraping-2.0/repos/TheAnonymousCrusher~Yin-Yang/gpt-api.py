# GPT
import openai
import requests
import json


# x str 'content 3 sentences from body'
# x =
openai.api_key = "api key"

completion = openai.ChatCompletion.create(
  model = "gpt-3.5-turbo", #GPT model
   messages=[
        {"role": "system", "content": "You are a philosopher"},
        {"role": "user", "content": "What is life?"}
        ]
)
# system gives context about user to assistant
# user is the prompt
