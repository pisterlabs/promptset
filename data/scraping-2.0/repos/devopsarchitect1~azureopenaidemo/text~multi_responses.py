import os
import openai

openai.api_type = "azure"
openai.api_base = "https://neyahopenai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  engine="text_demo",
  messages = [{"role":"system","content":"You are a cake baker"},
              {"role":"user","content":"Get me the recipe to make a cake"}],
  n=3
)

for choice in response.choices:
  print(choice.message.content)