from pathlib import Path
import json
import os
import re

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import openai
import typer

from src.data import load_data, sample_data

app = typer.Typer()


def create_prompt(dataset, id2label):
    labels = [id2label[label] for label in set(dataset["label"])]
    prompt = "Classify the sentence as one of {}\n".format(",".join(labels))

    for example in dataset:
        text, label = example["text"], example["label"]
        label = id2label[label]

        prompt += f"Text:{text}\n"
        prompt += f"Label:{label}\n\n"

    return prompt


def gpt3(model, prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=0.75, max_tokens=10
    )
    return response["choices"][0]["text"].split("\n\n")[0]


@app.command()
def evaluate(
    data_path="ag_news",
    model="text-davinci-002",
    n_shot: int = 8,
    n_folds: int = 5,
    test_size: int = 100,
    results_dir="results",
):
    dataset = load_data(data_path)

    id2label = {
        i: label for i, label in enumerate(dataset["train"].features["label"].names)
    }

    results = []
    for fold in range(n_folds):
        sample_dataset = sample_data(dataset, n_shot, test_size, fold)
        test_dataset = sample_dataset["test"]

        prompt = create_prompt(sample_dataset["train"], id2label)
        if len(prompt.split()) > 2500:
            continue

        y_test = []
        y_pred = []
        for i, example in enumerate(tqdm(test_dataset)):
            text = example["text"]

            prompt += f"Text:{text}\nLabel:"
            if len(prompt.split()) > 2600:
                continue

            try:
                pred = gpt3(model, prompt)
            except:
                print("Exceeded #tokens or errored")
                continue

            y_pred.append(pred)
            y_test.append(id2label[test_dataset["label"][i]])

        score = accuracy_score(y_test, y_pred)

        print(f"Skipped: {test_size-len(y_pred)}")
        print(score)

        results.append({"fold": fold, "metrics": {"accuracy": score}})

    if "/" in data_path:
        # if dataset comes from custom account, keep just the name
        data_path = data_path.split("/")[1]
    results_path = os.path.join(results_dir, data_path, "gpt3")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
