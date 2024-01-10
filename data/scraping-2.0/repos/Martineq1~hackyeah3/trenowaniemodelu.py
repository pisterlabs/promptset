import openai
import requests
import json
import os

API_KEY = "***" 
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
FolderPath2 = "F:\Git\hackyeah3\plikitreningowe\\trening1.json"
datatraining = ""

openai.api_key = API_KEY

training_file_name = FolderPath2 

training_response = openai.File.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
training_file_id = training_response["id"]


print("Training file id:", training_file_id)

suffix_name = "chatner-bot"

response = openai.FineTuningJob.create(
    training_file=training_file_id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)

job_id = response["id"]

print(response)

response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])

    response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print(response)
print("\nFine-tuned model id:", fine_tuned_model_id)
