import os
import openai
# openai.api_key = ""

def extract_code_chatgpt(text):
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You will act as a code extracter. I will give you a text which contains both code and explanation. I want you to give me only the code and only the code. Not even tell me in which language it is written. Just the code and nothing else. The code will be directly executed."},
      {"role": "user", "content":text}
    ],
    temperature = 0
  )
  code = completion.choices[0].message.content
  return code
