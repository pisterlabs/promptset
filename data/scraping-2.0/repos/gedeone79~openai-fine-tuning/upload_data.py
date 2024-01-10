import os
import openai

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

r  = openai.File.create(
  file=open("model.jsonl", "rb"),
  purpose='fine-tune'
)

print(r)
