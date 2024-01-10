from openai import OpenAI

client = OpenAI()

client.fine_tuning.jobs.create(training_file="./train.jsonl", model="gpt-3.5-turbo")
