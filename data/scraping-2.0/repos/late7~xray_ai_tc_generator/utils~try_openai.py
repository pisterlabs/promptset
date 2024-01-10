import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY") # set your API key using the 'export OPENAI_API_KEY=xyz........' environment variable

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {
      "role": "system",
      "content": "Define 1 Test Cases for the provided requirement specification. Response must be simple list with Description and test steps. No additional instructions"
    },
    {
      "role": "user",
      "content": "Data Precision and Accuracy. Requirement description: Connectivity Unit shall not miss more than 1 message per 100000 transferred ones. Missed messages shall be logged into errors list."
    }
  ],
  temperature=1,
  max_tokens=1706,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(response)