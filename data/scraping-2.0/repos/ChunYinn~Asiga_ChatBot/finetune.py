#still testing, not enough buget to train

import os
import openai
openai.api_key = "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"
openai.File.create(
  file=open("train_data.jsonl", "rb"),
  purpose='fine-tune'
)


openai.FineTuningJob.create(training_file="file-abc123", model="gpt-3.5-turbo")

# Retrieve the state of a fine-tune
openai.FineTuningJob.retrieve("ft-abc123")

