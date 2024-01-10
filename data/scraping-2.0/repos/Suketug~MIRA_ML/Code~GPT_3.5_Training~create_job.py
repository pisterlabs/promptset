import os
import openai
import time


openai.api_key = "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"

# Upload the training and validation data to OpenAI
train_file = openai.File.create(file=open("Data/train_data.jsonl", "rb"), purpose='fine-tune')
val_file = openai.File.create(file=open("Data/val_data.jsonl", "rb"), purpose='fine-tune')

# Wait for a few seconds to let OpenAI process the files
time.sleep(180)

# Get the file IDs from the upload responses
train_file_id = train_file['id']
val_file_id = val_file['id']

# Create a fine-tuning job
fine_tuning_job = openai.FineTuningJob.create(
    training_file=train_file_id, 
    validation_file=val_file_id, 
    model="gpt-3.5-turbo-0613"
)
