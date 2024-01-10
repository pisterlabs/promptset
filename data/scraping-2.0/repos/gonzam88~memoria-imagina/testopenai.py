import os
from openai import OpenAI

OPENAI_API_KEY="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"
client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, experienced documentary filmmaker with some traits of Herzog and Wong Kar Wai."},
    {"role": "user", "content": "Write in first person about your childhood memory. be specific, extensive and honest with your feeling. The topic is applying cream"}
  ]
)

print(completion.choices[0].message)