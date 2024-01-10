import os
from datetime import datetime
import openai

openai.api_key = os.environ.get('OPEN_API_KEY')

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="poem in kiswahili",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

with open(f'mashairi/{datetime.today().strftime("%d-%m-%y")}.txt', 'w+', encoding="utf-8") as file:
    file.write(response["choices"][0]["text"].strip())
