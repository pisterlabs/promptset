# this python scripts uploads fine-tuning datasets in OpenAI API file system

import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("openaiAPI"))

client.files.create(
  file=open("Approach1_train_dataset.jsonl", "rb"),
  purpose="fine-tune"
)

client.files.create(
  file=open("Approach2_train_dataset.jsonl", "rb"),
  purpose="fine-tune"
)

client.files.create(
  file=open("Approach3_train_dataset.jsonl", "rb"),
  purpose="fine-tune"
)

print("Just created the data file; proceed with fine-tuning\n")

