#api_key = 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
#pip install openai
#npm install openai
import os
import openai
openai.api_key = "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"
#list of names of different models available for OpenAI gpt-3
# print(openai.Model.list())

answer = openai.Completion.create(
  model="text-davinci-003",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)
print(answer)