#!/usr/bin/env python
try:
  import os
  import openai

  openai.api_key = os.getenv("OPENAI_API_KEY")
  gpt_prompt = "Correct this to standard English:\nPablo no went to the store.\n"

  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  
  print("Q: " + gpt_prompt)
  print("A: " + response['choices'][0]['text'].strip())

except Exception as e: 
  print(e)
