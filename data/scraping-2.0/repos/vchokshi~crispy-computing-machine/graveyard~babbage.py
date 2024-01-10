import os
import openai

import pprint

pp = pprint.PrettyPrinter(indent=1)

openai.api_key = os.getenv("OPENAI_API_KEY")

print(f'Type in any incomplete sentence and let me finish it for you!')
sentence = input(">")

response = openai.Completion.create(
  model="text-ada-001",
  prompt=f'Finish this sentence.\n\n{sentence}',
  temperature=0.79,
  max_tokens=128,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["."]
)
#pp.pprint(response)
finish = response.choices[0].text
print(f'{sentence} {finish}.')
