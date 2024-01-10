#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://cog-cl7i2ktocqmee.openai.azure.com/"
openai.api_version = "2022-12-01"
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "9e5688666bc04668b80aeaf49bccda16"

response = openai.Completion.create(
  engine="code",
  prompt="Generate a python function to reverse a string.The function should be an optimal solution in terms of time and space complexity.",
  temperature=0.2,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  best_of=1,
  stop=["#"])

print(response.choices[0].text)




