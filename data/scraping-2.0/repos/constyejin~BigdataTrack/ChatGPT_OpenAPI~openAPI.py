import os
import openai

openai.organization = 'org-ZfFzhcXEEd613Acj1J0YnVQ5'
openai.api_key = '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'

input_text = "인공지능에 대해서 알려 줘"

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[{"role": "user", "content": input_text}]
)

output_text = response['choices'][0]['message']['content']
output_text

response