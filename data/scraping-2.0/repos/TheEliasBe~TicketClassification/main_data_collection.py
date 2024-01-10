import os
import subprocess
import time
from datetime import datetime

import pandas as pd
import openai
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score

import wandb
import yaml

hyper_param_grid = {
    "n_epochs": [8, 16, 32],
    "training_samples": [200, 400, 600],
    "learning_rate_multiplier": [0.05, 0.1, 0.5],
    "prompt_loss_weight": [0.01, 0.1]
}

with open("conf/base/parameters.yml") as f:
    parameters = yaml.safe_load(f)


def fine_tune_gpt_model(n_epochs, training_samples, learning_rate_multiplier, prompt_loss_weight, model_id=None):
    wandb.init()
    openai.api_key = parameters["OPENAI_API_KEY"]
    subprocess.run(["openai", "tools", "fine_tunes.prepare_data", "-f", "data/05_model_input/2022.jsonl", "-q"])
    if not model_id:
        df = pd.read_json("data/05_model_input/2022.jsonl", lines=True)
        df.sample(n=training_samples).to_json("data/05_model_input/2022.jsonl", lines=True, orient="records")

        response = openai.File.create(file=open("data/05_model_input/2022_prepared_train.jsonl"), purpose="fine-tune")

        params = {
            "model": "ada",
            "n_epochs": n_epochs,
            "learning_rate_multiplier": learning_rate_multiplier,
            "prompt_loss_weight": prompt_loss_weight,
        }

        wandb.log({"target": "Produkt Label"})
        wandb.log({"n_epochs": n_epochs})
        wandb.log({"model": params["model"]})
        wandb.log({"learning_rate_multiplier": params["learning_rate_multiplier"]})
        wandb.log({"prompt_loss_weight": params["prompt_loss_weight"]})

        fine_tune_job = openai.FineTune.create(
            training_file=response.id,
            model=params["model"],
            # n_epochs=params["n_epochs"],
            # learning_rate_multiplier=params["learning_rate_multiplier"],
            # prompt_loss_weight=params["prompt_loss_weight"]
        )
        print("### CREATED JOB")
        print(fine_tune_job)

        # wait for fine tuning to finish
        while True:
            events = openai.FineTune.list_events(id=fine_tune_job.id)
            print(f"{datetime.now().strftime('%H:%M:%S')}: Fine-tuning job status: {events['data']}")
            print("Events.dat", events["data"])
            status = openai.FineTune.retrieve(fine_tune_job.id).status
            print(f"{datetime.now().strftime('%H:%M:%S')}: Fine-tuning job status: {status}")

            if len(events) == 0:
                print("No events yet", events)
            else:
                if "succeeded" in events["data"][-1]["message"]:
                    try:
                        cost = float(events[1]["message"][-4:])
                    except:
                        cost = 0
                    break

            time.sleep(60)

        retrieve_response = openai.FineTune.retrieve(fine_tune_job.id)
        fine_tuned_model = retrieve_response.fine_tuned_model
    else:
        fine_tuned_model = openai.FineTune.retrieve(id=model_id).fine_tuned_model

    validation_df = pd.read_json(path_or_buf="data/05_model_input/2022_prepared_valid.jsonl", lines=True)

    def get_classification(prompt):
        print("Prompt", type("prompt"), prompt)
        answer = openai.Completion.create(
            model=fine_tuned_model,
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=2,
        )
        return answer['choices'][0]['text']

    validation_df['classification'] = validation_df['prompt'].apply(get_classification)
    print(validation_df.head())
    print(classification_report(validation_df['completion'], validation_df['classification']))

    # log metrics
    wandb.log({"accuracy": accuracy_score(validation_df['completion'], validation_df['classification'])})
    wandb.log(
        {"precision": precision_score(validation_df['completion'], validation_df['classification'], average='macro')})
    wandb.log({"recall": recall_score(validation_df['completion'], validation_df['classification'], average='macro')})

    # log hyperparameters
    wandb.log({"f1": f1_score(validation_df['completion'], validation_df['classification'], average='macro')})
    wandb.log({"training_examples": len(
        pd.read_json(path_or_buf="data/05_model_input/2022_prepared_train.jsonl", lines=True))})
    wandb.log({"model": params["model"]})
    wandb.log({"model_id": str(fine_tuned_model)})
    wandb.log({"cost": cost})
    wandb.log({"vocabulary_size": parameters["VOCAB_THRESHOLD"]})

    # save predictions
    print(f"Saving to {wandb.run.name}.csv")
    validation_df.to_csv(f"data/07_model_output/{wandb.run.name}.csv", index=False)

    wandb.finish()

    # delete file used for training and validation
    os.remove("data/05_model_input/2022_prepared_train.jsonl")
    os.remove("data/05_model_input/2022_prepared_valid.jsonl")
    os.remove("data/05_model_input/2022.jsonl")


fine_tune_gpt_model(
    n_epochs=hyper_param_grid["n_epochs"][0],
    training_samples=hyper_param_grid["training_samples"][0],
    learning_rate_multiplier=hyper_param_grid["learning_rate_multiplier"][0],
    prompt_loss_weight=hyper_param_grid["prompt_loss_weight"][0],
    # model_id="ft-UjBgWXi0HRNa8drkjwKv6PXO"
)