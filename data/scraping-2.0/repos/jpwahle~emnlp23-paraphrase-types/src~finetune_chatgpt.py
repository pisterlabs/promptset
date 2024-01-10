import argparse
import os

import openai


def parse_input_data():
    parser = argparse.ArgumentParser(description="Parse input data argument.")
    parser.add_argument("--input_data", type=str, help="Input data for fine-tuning.")
    args = parser.parse_args()
    return args.input_data


if __name__ == "__main__":
    input_data = parse_input_data()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    file = openai.File.create(file=open(input_data, "rb"), purpose="fine-tune")
    job = openai.FineTuningJob.create(
        training_file=file.id, model="gpt-3.5-turbo", hyperparameters={"n_epochs": 1}
    )
    openai.FineTuningJob.list_events(id=job.id, limit=10)
