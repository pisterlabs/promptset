#!/usr/bin/python3
import os
import sys
import openai

question       = ' '.join(sys.argv[1:])

openai.api_key = os.getenv("OPENAI_API_KEY")

#
#   Set model
#  (Find a list of models at https://api.openai.com/v1/models)
#
model = 'text-davinci-002'
      # 'gpt-3.5-turbo'
      # 'gpt-3.5-turbo-0301'

resp = openai.Completion.create(
  model             = model,
  prompt            = question + '?',
  temperature       =   0,
  max_tokens        = 100,
  top_p             =   1,
  frequency_penalty = 0.0,
  presence_penalty  = 0.0,
  stop              =["?"]
)

print(resp["choices"][0]["text"])
