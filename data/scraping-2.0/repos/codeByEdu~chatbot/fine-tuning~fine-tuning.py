import openai
import os
  
api_key = os.environ["API_KEY"]

openai.api_key = api_key

with open("fine-tuning\interview-data.jsonl", encoding="utf-8") as file:
  response = openai.File.create(
    file=file,
    purpose='fine-tune'
  )

file_id = response['id']
print(f"ID: {file_id}")

#Usando o File ID
file_id =  file_id
model_name = "gpt-3.5-turbo"

response = openai.FineTuningJob.create(
  training_file=file_id,
  model=model_name
)

job_id = response['id']
print(f"job id={job_id}")