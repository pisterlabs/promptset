import typer
import os
import openai
import json
openai.api_key = os.getenv("OPENAI_API_KEY")


def _get_file_ids():
    with open("output/fine_tune_file_ids.json", "r") as fin:
        file_ids = json.load(fin)
        return file_ids['training_file_id'], file_ids['validation_file_id']


def create_fine_tune_job(model: str, suffix: str):
    training_file_id, validation_file_id = _get_file_ids()

    fine_tuning_job = openai.FineTuningJob.create(
        model=model,
        training_file=training_file_id,
        validation_file=validation_file_id,
        suffix=suffix,
        hyperparameters={
            "n_epochs": 5
        }
    )

    return fine_tuning_job["id"]


def monitor_fine_tune_job(job_id):
    import time

    while True:
        fine_tuning_status = openai.FineTune.get_status(job_id)
        status = fine_tuning_status["status"]
        print(f"Fine-tuning job status: {status}")

        if status in ["completed", "failed"]:
            break

        time.sleep(60)


if __name__ == '__main__':
    typer.run(create_fine_tune_job)

