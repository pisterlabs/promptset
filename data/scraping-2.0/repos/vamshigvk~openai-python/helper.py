import os
import openai


api_key='<your_api_key>'

# 
def qna(prompt):
  openai.api_key = api_key

  completions = openai.Completion.create(
    prompt=prompt,
    engine="text-davinci-002", # text-curie-001, text-babbage-001, text-ada-001, text-davinci-001, davinci-instruct-beta, davinci, curie, babbage, ada
    max_tokens=100
  )

  completion = completions.choices[0].text
  print(f'completion: {completions}')
  return completion


