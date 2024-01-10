import os

import openai

PROMPT = "tippu sulthan in madurai temple"

openai.api_key = "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"

response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size="1024x1024",
)

print(response["data"][0]["url"])