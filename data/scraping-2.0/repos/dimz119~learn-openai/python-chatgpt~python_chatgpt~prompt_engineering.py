import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Provide the korean history question including 4 possible choices " +
          "and provide the correct answer\n the answer format start with 'Answer:'",
  max_tokens=300,
  temperature=0.7
)
# print(response)
print(response['choices'][0]['text'])
