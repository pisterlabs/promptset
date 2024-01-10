import os
import openai as op
from .BaseModel import BaseModel


op.api_key = "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"

response = op.Completion.create(model="gpt-3.5-turbo", prompt="Say this is a test", temperature=0, max_tokens=7)

print(response)

