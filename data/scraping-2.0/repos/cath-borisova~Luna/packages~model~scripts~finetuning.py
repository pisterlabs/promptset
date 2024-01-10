import os
import openai
openai.api_key = "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"

response = openai.File.create(
  file=open("../data/finetuning_data.jsonl", "rb"),
  purpose='fine-tune'
)
openai.FineTuningJob.create(training_file=response.id, model="gpt-3.5-turbo")
