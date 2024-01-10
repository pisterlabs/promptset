import openai

job_id = "ftjob-aZIxmElrMHMAf6avscP782lA"

response = openai.FineTuningJob.retrieve(job_id)

status = response["status"]

if status == "succeeded":
  print("Training succeeded!")
  model_id = response["fine_tuned_model"]
  print("Fine-tuned model ID:", model_id)
else:
  print("Job status:", status)