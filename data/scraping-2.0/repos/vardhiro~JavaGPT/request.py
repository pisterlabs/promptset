import json
import sys
import os
import openai


# Load your API key from an environment variable or secret management service
openai.api_key = "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"

response = openai.Completion.create(model="text-davinci-003", prompt=sys.argv[1], temperature=0, max_tokens=700)

print(response["choices"][0]["text"])
