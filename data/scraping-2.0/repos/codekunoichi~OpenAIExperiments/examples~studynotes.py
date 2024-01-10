import os
import openai
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="What are 5 key points I should know when starting with natural language processing?",
  temperature=0.3,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

print(type(response))
print(response)
# Parsing JSON data using json.loads()
parsed_data = json.loads(str(response))

print(parsed_data["choices"][0]["text"])
