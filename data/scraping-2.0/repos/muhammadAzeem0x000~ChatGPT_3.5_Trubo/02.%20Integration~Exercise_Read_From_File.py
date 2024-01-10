import os
import openai
import secret
# WRITE YOUR CODE HERE

openai.api_key = secret.api_key

def chatgpt_code_review(file_path):

  with open(file_path, 'r') as file:
    code = file.read()
  response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": code}
        ]
  )
  return response['choices'][0]['message']['content'].strip()

file_path = "temp.py"
print(chatgpt_code_review(file_path))