import openai

openai.api_key_path = ".openai-api-key"
openai.File.create(
    file=open("data/generated_data.jsonl", "rb"), purpose="fine-tune"
)

print(openai.File.list())

openai.FineTuningJob.create(
    training_file="file-xYgYu76YizAh1t9hrAxTPeYc", model="gpt-3.5-turbo"
)
#
# # List 10 fine-tuning jobs
print(openai.FineTuningJob.list(limit=10))
