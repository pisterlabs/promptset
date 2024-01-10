import fire
import os
import openai
import time

def submit_file_tune_job(file_path, model="gpt-3.5-turbo"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    print(f"Uploading file {file_path}")
    file_info = openai.File.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )
    print(f"Fine tune file info: {file_info}")

    while file_info.status != "processed":
        print("Waiting for file to be processed")
        time.sleep(5)
        file_info = openai.File.retrieve(file_info.id)
        print(file_info)
    
    print("Submitting finetune job")
    ft_job_info = openai.FineTuningJob.create(training_file=file_info.id, model=model)
    print(ft_job_info)


if __name__ == "__main__":
    fire.Fire(submit_file_tune_job)