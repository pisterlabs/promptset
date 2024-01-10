import os
import openai  # !pip install openai==0.27.9
from time import sleep

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
openai.api_key = os.environ["OPENAI_API_KEY"]


def main():
    res = openai.File.create(
        file=open("./data.jsonl", "r"),
        purpose='fine-tune'
    )

    file_id = res["id"]

    job = openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo"
    )

    job_id = job["id"]

    while True:
        res = openai.FineTuningJob.retrieve(job_id)
        if res["finished_at"] != None:
            break
        else:
            print(".", end="")
            sleep(100)

    ft_model = res["fine_tuned_model"]
    return ft_model
