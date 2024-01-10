import pandas as pd
from collections import Counter
import json
import os
import tiktoken
import openai
import backoff
import subprocess
from dotenv import load_dotenv
load_dotenv()
import time

openai.organization = os.getenv("ORG_ID")


def fineTune():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    file_upload = openai.File.create(file=open("finetune/train_data.jsonl", "rb"), purpose="fine-tune")
    print("Uploaded file id", file_upload.id)

    while True:
        print("Waiting for file to process...")
        file_handle = openai.File.retrieve(id=file_upload.id)
        if len(file_handle) and file_handle.status == "processed":
            print("File processed")
            break
        time.sleep(3)


    job = openai.FineTuningJob.create(training_file=file_upload.id, model="gpt-3.5-turbo")

    while True:
        print("Waiting for fine-tuning to complete...")
        job_handle = openai.FineTuningJob.retrieve(id=job.id)
        if job_handle.status == "succeeded":
            print("Fine-tuning complete")
            print("Fine-tuned model info", job_handle)
            print("Model id", job_handle.fine_tuned_model)
            break
        time.sleep(3)

def test():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    model_id = "ft:gpt-3.5-turbo:my-org:custom_suffix:id"

    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {
                "role": "system",
                "content": "As a response, provide the following fields in a JSON dict: name, handle, age, hobbies, email, bio, location, is_blue_badge, joined, gender, appearance, avatar_prompt, and banner_prompt.",
            },
            {
                "role": "user", 
                "content": "Generate details of a random Twitter profile."
            },
        ],
    )

if __name__ == '__main__':
    fineTune()
    test()