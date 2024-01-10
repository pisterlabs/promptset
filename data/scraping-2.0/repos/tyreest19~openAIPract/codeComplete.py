import os
import openai

api_key = "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"
openai.api_key = api_key

prompt_text = """ Using python I want you to import the selenium webdriver library, set the webdriver's browser as google chrome, tell the webdriver to visit get google.com and then make a page object model to test google.com search feature """

response = openai.Completion.create(
  model="text-davinci-002",
  prompt= prompt_text,
  temperature=0,
  max_tokens=1000,
  n=5

)

for choice in response['choices']:
    print(choice['text'])
