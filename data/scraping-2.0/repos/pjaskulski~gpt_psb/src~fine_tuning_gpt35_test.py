""" function calling test """
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import openai


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY


#psb_data = Path("..") / "data_fine_tuning" / "psb_data.jsonl"
psb_data = Path("..") / "data_fine_tuning" / "ahp.jsonl"

file_data = openai.File.create(
    file=open(psb_data, 'r', encoding='utf-8'),
    purpose='fine-tune',
)

print('Wczytywanie pliku...')
time.sleep(30)  # czas na wczytanie pliku przez OpenAI

file_id = file_data['id']

ft_job = openai.FineTuningJob.create(
    training_file=file_id,
    model='gpt-3.5-turbo',
)

job_id = ft_job['id']

model_id = None
print('Proces trwa...')

while True:
    job_status = openai.FineTuningJob.retrieve(job_id)
    if job_status['finished_at'] is not None:
        break

    print('.', end='')
    time.sleep(30)

model_id = job_status['fine_tuned_model']
print()
print(model_id)
