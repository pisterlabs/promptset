import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "user", "content": "Tell me a joke about a bored computer scientist who is writing an app to ask a language model to write some proofs"},
  ]
)

print(response)
