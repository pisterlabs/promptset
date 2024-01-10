import os
import time

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# oa-MINE CHATGPT DESCRIPTION WITH EXAMPLE VALUES
training_file_id = '' # <file_id>
not_started = True
dataset = 'oa-mine' # ae-100k
while(not_started):
    try:
        openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo-0613", suffix=f"des_ex-{dataset}")
        not_started = False
    except openai.error.InvalidRequestError as e:
        print(e)
        print("Waiting for 60 seconds...")
        time.sleep(60)
    except openai.error.RateLimitError as e:
        print(e)
        jobs = openai.FineTuningJob.list(limit=10)
        print('Active jobs:')
        for job in jobs.data:
            if job.status == 'running':
                print(job)
        print("Waiting for 600 seconds...")
        time.sleep(600)
