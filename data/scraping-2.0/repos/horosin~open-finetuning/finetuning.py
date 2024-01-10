import os
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

file_upload = openai.File.create(file=open("mydata.jsonl", "rb"), purpose="fine-tune")
print("Uploaded file id", file_upload.id)

while True:
    print("Waiting for file to process...")
    file_handle = openai.File.retrieve(id=file_upload.id)
    if len(file_handle) and file_handle.status == "processed":
        print("File processed")
        break
    time.sleep(3)


job = openai.FineTuningJob.create(training_file=file_upload.id, model="gpt-3.5-turbo")

while True:
    print("Waiting for fine-tuning to complete...")
    job_handle = openai.FineTuningJob.retrieve(id=job.id)
    if job_handle.status == "succeeded":
        print("Fine-tuning complete")
        print("Fine-tuned model info", job_handle)
        print("Model id", job_handle.fine_tuned_model)
        break
    time.sleep(3)
