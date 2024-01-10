import openai, json, time

openai.api_key = "hidden"

file = openai.File.create(
  file=open('fine-tuning/training_data.jsonl'),
  purpose='fine-tune',
)

print("File Content:")
openai.File.retrieve("file-tsGG0BVPDGKlOJAihYI42Y85")
	
job = openai.FineTuningJob.create(
  training_file='your_file_id',
  model='gpt-3.5-turbo',
)

print("Job ID:" + job.id)

openai.FineTuningJob.retrieve(job.id)

print(file)


