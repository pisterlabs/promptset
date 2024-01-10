import openai
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)
openai.api_key = os.environ.get("OPENAI_API_KEY")

def create_file(data_path):
    response = openai.File.create(file=open(data_path, "rb"), purpose="fine-tune")
    file_id = response["id"]
    print("file_id: ", file_id)
    # file_id.txtというファイル名で保存
    with open("file_id.txt", mode="w") as f:
        f.write(f"コピーして↓\n{file_id}")
    return file_id

def fine_tune(file_id):
    response_job = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
    job_id = response_job["id"]
    print("job_id: ", job_id)
    # job_id.txtというファイル名で保存
    with open("job_id.txt", mode="w") as f:
        f.write(f"コピーして↓\n{job_id}")
    return job_id

