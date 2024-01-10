import openai
from dotenv import load_dotenv
import os
from time import sleep
from datetime import datetime

load_dotenv('../.env')
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
openai.api_key = OPEN_AI_API_KEY

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'a', encoding='utf-8') as outfile:
        outfile.write(content)


def upload_gpt():
    with open("./src/fine_tune.jsonl", "rb") as file:
        response = openai.File.create(
            file=file,
            purpose='fine-tune'
        )

    file_id = response['id']
    print(f"File uploaded successfully with ID : {file_id}")
    return file_id


def tuning_gpt(file_id):
    model_name = "gpt-3.5-turbo"
    response = openai.FineTuningJob.create(
        training_file=file_id,
        model=model_name
    )

    job_id = response['id']
    print(f"Fine-tuning job created successfully with ID : {job_id}")

    return job_id


def retrieve_gpt(job_id):
    while True:
        res = openai.FineTuningJob.retrieve(job_id)
        if res['status'] != "running":
            f = open("./src/fine_tune_model.txt", 'a')
            f.write(str(datetime.now()) + '\n')
            f.write(str(res))
            f.write('\n')
            f.close()
            break
        else:
            print(".", end="", flush=True)
            sleep(100)


uploaded_file_id = upload_gpt()
sleep(100)
uploaded_job_id = tuning_gpt(uploaded_file_id)
retrieve_gpt(uploaded_job_id)