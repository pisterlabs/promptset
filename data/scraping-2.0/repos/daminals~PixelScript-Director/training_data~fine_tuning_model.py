import openai
import json
import os
import random

openai.api_key = os.environ["OPENAI_API_KEY"]
training_file = openai.File.create(
  file=open("fine_tune_model.jsonl", "rb"),
  purpose='fine-tune',
  user_provided_filename="ams560-fine_tune_model"
)

print(training_file)

job = openai.FineTuningJob.create(
    training_file=training_file["id"], model="gpt-3.5-turbo-1106")
