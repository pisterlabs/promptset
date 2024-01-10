import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "RightOnTrek is a friendly encyclopedia for hikers in the US."},
    {"role": "user", "content": ""}
  ]
)
print(completion.choices[0].message)
