import openai
import os
import typer
import json
openai.api_key = os.getenv("OPENAI_API_KEY")

def fine_tune_status(output_file: str):
    fine_tune_jobs = openai.FineTuningJob.list()['data']
    last_job = max(fine_tune_jobs, key=lambda x: x.created_at)
    last_job_dict = last_job.to_dict()
    print(last_job_dict)
    json.dump(last_job_dict, open(output_file, 'w'))

if __name__ == '__main__':
    typer.run(fine_tune_status)

