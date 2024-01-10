import openai
import os
import json
#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
openai.api_key = "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"
model_engine = "davinci:ft-personal-2023-04-09-03-29-53"
prompt = "ce?"
max_tokens = 200
response = openai.Completion.create(
    model=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    n=1,
    stop=".\n",
    temperature=0.5,
)


for choice in response.choices:
    print(choice.text)
