import os
import openai

openai.api_type = "azure"
openai.api_base = "https://neyahopenai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  engine="text_demo",
  messages = [{"role":"user","content":"Tell a gun joke"}]
)

print(response)
print(response["choices"][0]["message"]["content"])