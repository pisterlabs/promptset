from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
fine_tuning_job_id = os.getenv('CURRENT_FINE_TUNING_JOB_ID')
client = OpenAI(api_key=key)
response = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
status = response.status

responseEvents = client.fine_tuning.jobs.list_events(fine_tuning_job_id=fine_tuning_job_id, limit=10)
print(f"status code: {status}, events {responseEvents}")
