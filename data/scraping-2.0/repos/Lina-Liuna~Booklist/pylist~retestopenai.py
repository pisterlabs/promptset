from openai import OpenAI

client = OpenAI(api_key='lina-key')

for job in client.fine_tuning.jobs.list(limit=2):
    print(job)