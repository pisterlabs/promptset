import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = "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"
i = "openai is a"
response = openai.Completion.create(engine="curie", prompt=i, max_tokens=50)
print("[" + i + "]" + response["choices"][0]["text"])

