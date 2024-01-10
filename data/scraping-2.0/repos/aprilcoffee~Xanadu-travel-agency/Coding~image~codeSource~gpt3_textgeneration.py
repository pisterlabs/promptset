
import api_key

openai_key = api_key.openai_key()

import os
import openai
openai.api_key = openai_key

prompting_text = '''
write a long postcard to friends associate with following notes
beach, sunset
'''


import translators as ts


response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompting_text,
  temperature=1,
  max_tokens=256,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=['/n/n/n/n']
)

print(response.choices[0].text)
t = ts.google(response.choices[0].text,from_language='en',to_language='ru')
print(t)
