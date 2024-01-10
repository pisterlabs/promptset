# _____________________ A LEGACY -- OLD OPENAI MODELS

# ON COMMAND LINE
# Linux/MacOS: set/export OPENAI_API_KEY = "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"
# windows: $env:OPENAI_API_KEY="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"

# data preparation: openai tools fine_tunes.prepare_data -f data.jsonl
# creating the FT model: openai api fine_tunes.create -t data_prepared.jsonl -m davinci


# _____________________ Innovation -- NEW OPENAI MODELS
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# STEP-1: uploading the training file
data = openai.File.create(
    file=open("..\Datasets\Raw\data_ccf.jsonl", "rb"),
    purpose='fine-tune'
)
print(data.id) # ex: file-rjwhrIOw7EdTbWhFqVlYYGlc

# STEP-2: creating the model
resp = openai.FineTuningJob.create(training_file=data.id, model="gpt-3.5-turbo")
print(resp)  
#ex1 fine-tune job: ftjob-FrCdJ2SQcVLl9HttPhkk7pZL


# STEP-3: Using the model

# List 10 fine-tuning jobs
jobs = openai.FineTuningJob.list(limit=10)
print (jobs) #--> get the fine-tuning job id (data.id)

# Retrieve the state of a fine-tune job
print (openai.FineTuningJob.retrieve("ftjob-18xMJG80MoGVe93PukXnLxmM"))
