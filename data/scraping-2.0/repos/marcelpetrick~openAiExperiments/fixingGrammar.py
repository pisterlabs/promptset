import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  #model="gpt-4", # does not work - completion not possible
  prompt="Correct this to standard English:\n\nShe do not went to the her owns garden.",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print(f"response: {response}")
