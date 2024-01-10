import random
import os
import openai


model="gpt-4"
max_tokens=2048
temperature=0.5

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

def GPT4_openai(messages_list):
    response = openai.ChatCompletion.create(
    model=model,
    max_tokens=max_tokens,
    temperature=temperature,
    messages = messages_list)
    return response['choices'][0]['message']['content']

# system instruction
def Agent_Create(Prompt_text):
  messages_list=[{"role": "system", "content": Prompt_text}]
  return messages_list

# chat flow
def GPT_Chat(messages_list, new_message, role):
    # append new message
    messages_list.append({"role": role, "content": new_message})
    return messages_list