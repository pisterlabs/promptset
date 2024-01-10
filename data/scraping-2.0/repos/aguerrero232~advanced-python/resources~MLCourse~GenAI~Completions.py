import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

prefix = input("Enter some text to complete: ")

completion = openai.Completion.create(
  model="text-davinci-003",
  prompt=prefix,
  max_tokens=100,
  temperature=0
)

print("\n" + completion.choices[0].text)
