import typer
import openai
import os
# from openai import File

def fine_tune(output_file: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    fine_tune_jobs = openai.FineTuningJob.list()['data']
    last_job = max(fine_tune_jobs, key=lambda x: x.created_at)
    results_file_id = last_job['result_files'][0]
    content = openai.File.download(results_file_id)
    with open(output_file, 'wb') as file:
        file.write(content)

if __name__ == "__main__":
    typer.run(fine_tune)
