import os
import time
import argparse

import openai
from openai import OpenAI
from validate_json import validate_json


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def launch_training(finetuning_train_path: str, finetuning_val_path: str = None) -> None:
    validate_json(finetuning_train_path)
    if finetuning_val_path:
        validate_json(finetuning_val_path)

    # upload file
    with open(finetuning_train_path, "rb") as f:
        train_output = client.files.create(
            file=f,
            purpose="fine-tune",
        )
    train_file_id = train_output.id

    valid_file_id = None
    if finetuning_val_path:
        with open(finetuning_val_path, "rb") as f:
            val_output = client.files.create(
                file=f,
                purpose="fine-tune",
            )
        valid_file_id = val_output.id

    print("File uploaded. Launching training job with information : {}".format(train_output))

    # launch training
    while True:
        try:
            job_output = client.fine_tuning.jobs.create(
                training_file=train_file_id,
                validation_file=valid_file_id,
                model="gpt-3.5-turbo-1106",
                suffix="hiep",
            )
            print("Job output: {}".format(job_output))
            break
        except openai.BadRequestError:
            print("Waiting for file to be ready...")
            time.sleep(60)
    print(
        f"Training job {job_output.id} launched. You will be emailed when it's complete."
    )

    # # Retrieve the state of a fine-tune
    # retrieve_training_info = client.fine_tuning.jobs.retrieve(job_output.id)
    # print("retrieve training info : {}".format(retrieve_training_info))
    # fine_tuned_model_id = retrieve_training_info["fine_tuned_model"]
    # print("Fine-tuned model id : {}".format(fine_tuned_model_id))


def parge_args():
    parser = argparse.ArgumentParser(description="Dataset preparation for fine-tuning")
    parser.add_argument(
        "-tp",
        "--train_path",
        type=str,
        default="datasets/finetuning_events_train.jsonl",
        help="Path to finetuning events training in .jsonl format",
    )
    parser.add_argument(
        "-vp",
        "--val_path",
        type=str,
        help="Path to finetuning events evaluate in .jsonl format",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parge_args()

    launch_training(args.train_path, args.val_path)
