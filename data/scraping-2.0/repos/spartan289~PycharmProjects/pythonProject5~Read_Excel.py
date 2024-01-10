import os
import openai
openai.api_key = "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"

user_input=''

with open('Book1.xlsx','r') as file:
    user_input=file.read()

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_input}
  ],
    temperature=0

)
ai_response = response.choices[0].message.content
print("Summarized text: \n\n,",ai_response)


