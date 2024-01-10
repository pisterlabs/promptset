import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.File.create(
  file=open('data/ubuntu/openai-finetune-ready-data-train.jsonl', 'rb'),
  purpose='fine-tune'
)
openai.File.create(
    file=open('data/ubuntu/openai-finetune-ready-data-test.jsonl', 'rb'),
    purpose='fine-tune'
)