import os
from openai import OpenAI
client = OpenAI()

job = client.fine_tuning.jobs.create(
  training_file="file-IGAJid6dRmDsw8YCGv2I17Db",
  model="gpt-3.5-turbo-1106"
)

print("Fine-tuning job started:", job.id)