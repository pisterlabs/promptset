from openai import OpenAI
import openai
from server_data.api_keys import openai_api

def explaine(content):
  client = OpenAI(api_key=openai_api)

  with open('utiles/quiz_yt_promt.txt') as f:
      promt = "".join(f.readlines())

  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": promt},
      {"role": "user", "content": content},
    ],
    temperature=0,
    max_tokens=1024
  )
  return response.choices[0].message.content