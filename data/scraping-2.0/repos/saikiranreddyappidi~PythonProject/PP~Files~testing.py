import os
import openai
OPENAI_API_KEY="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"
openai.api_key = OPENAI_API_KEY
arr=openai.Image.create(
  prompt="A cute baby puppy",
  n=2,
  size="1024x1024"
)
print(arr)