import openai
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

file_id = "file-MNFxwIAFg8uFbt7uMivKVKdE"
model_name = "ft:gpt-3.5-turbo-0613:personal::846t2axt"

response = openai.FineTuningJob.create(training_file=file_id, model=model_name)

job_id = response["id"]

print(f"Fine tunning job created with id: {job_id}")
