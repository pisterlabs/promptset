import os
import openai
import time


openai.debug = True
openai.log = "debug"

openai.api_key = os.getenv("OPENAI_API_KEY")

training_response = openai.File.create(
  file=open("gpt3.5_output.jsonl", "rb"),
  purpose='fine-tune'
)

training_file_id = training_response["id"]

print("Training file ID:", training_file_id)

response = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo")

print("Job ID:", response["id"])
print("Status:", response["status"])


time.sleep(10)

job_id = response["id"]

response = openai.FineTuningJob.retrieve(job_id)

print("Job ID:", response["id"])
print("Status:", response["status"])
print("Trained Tokens:", response["trained_tokens"])



response = openai.FineTuningJob.list_events(id=job_id, limit=50)

events = response["data"]
events.reverse()

for event in events:
    print(event["message"])



response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]

print("Fine-tuned model ID:", fine_tuned_model_id)