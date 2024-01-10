import openai
import os
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

training_file_response = openai.File.create(
    file=open("QA_Eminem_train.jsonl", "rb"), purpose="fine-tune"
)
training_file_id = training_file_response["id"]

validation_file_response = openai.File.create(
    file=open("QA_Eminem_valid.jsonl", "rb"), purpose="fine-tune"
)
validation_file_id = validation_file_response["id"]

time.sleep(5)
print("File uploading status:", validation_file_response["status"])

suffix_name = "Eminem"
response = openai.FineTuningJob.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)

job_id = response["id"]

print('Create job: ', response)

while response["status"] != "succeeded":
    response = openai.FineTuningJob.retrieve(job_id)
    time.sleep(5)
    print('Retrieve', response)

response = openai.FineTuningJob.retrieve(job_id)
fine_tuned_model_id = response["fine_tuned_model"]
print("\nFine-tuned model id:", fine_tuned_model_id)

response = openai.FineTuningJob.list_events(id=job_id, limit=50)
events = response["data"]
events.reverse()

for event in events:
    print(event["message"])