import openai
from openai.error import TryAgain, InvalidRequestError
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    fine_tune_jobs = openai.FineTuningJob.list()['data']
    last_job = max(fine_tune_jobs, key=lambda x: x.created_at)
    openai.FineTuningJob.cancel(last_job.id)
