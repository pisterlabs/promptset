import json
from time import sleep

import torch
from openai import OpenAI
import pandas as pd

import os

from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy

from civic.config import DATA_PROCESSED_DIR

client = OpenAI()

label_to_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def get_gpt_4_example(prepend_string, abstract):
    return f"""Metadata:\n\"{prepend_string}\"\nAbstract:\n:\"{abstract}\""""


def get_random_few_shot_examples(n_samples_per_evidence_level, is_test=False):
    if is_test:
        sampled_df = pd.read_csv(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4.csv")
        )
    else:
        df = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv"))
        sampled_df = pd.DataFrame()
        for value in df["evidenceLevel"].unique():
            sampled_rows = df[df["evidenceLevel"] == value].sample(
                n_samples_per_evidence_level
            )
            sampled_df = pd.concat([sampled_df, sampled_rows], axis=0)
    out = [
        {
            "example": get_gpt_4_example(
                sampled_df.iloc[i]["prependString"],
                sampled_df.iloc[i]["sourceAbstract"],
            ),
            "label": sampled_df.iloc[i]["evidenceLevel"],
        }
        for i in range(sampled_df.shape[0])
    ]
    return out


def _get_prompt_from_sample(sample):
    return [
        {
            "role": "user",
            "content": f"Extract the level of clinical significance from this combination of metadata and abstract:\n{sample['example']}",
        },
        {
            "role": "assistant",
            "content": f"""{sample['label']}""",
        },
    ]


def gpt4_query(examples, prompt):
    messages = (
        [
            {
                "role": "system",
                "content": "You are an expert on rare tumor treatments. In the following you will be presented with "
                + "abstracts from medical research papers. These abstracts deal with treatment approaches for rare "
                + "cancers as characterized by their specific genomic variants. Your job is to infer the level of "
                + "clinical significance of the investigations described in the abstract from the abstract and relevant"
                + 'metadata. The labels should range from "A" (indicating the strongest clinical significance) to '
                + '"E" (indicating the weakest clinical significance). You will answer machine-like with exactly one '
                + "character (the level of clinical significance you think is appropriate). You will be presented with "
                "examples.",
            },
            *examples,
            prompt,
        ],
    )
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview", messages=messages[0]
    )
    print(completion)
    projected_label = completion.choices[0].message.content
    return projected_label


def do_gpt4_evaluation(n_shots):
    f1_score = MulticlassF1Score(num_classes=5, average=None)
    macro_f1_score = MulticlassF1Score(num_classes=5, average="macro")
    micro_f1_score = MulticlassF1Score(num_classes=5, average="micro")
    accuracy = Accuracy(task="multiclass", num_classes=5)
    train_examples = [
        _get_prompt_from_sample(sample)
        for sample in get_random_few_shot_examples(n_shots)
    ]
    examples = sum(train_examples, [])
    test_examples = [
        _get_prompt_from_sample(sample)
        for sample in get_random_few_shot_examples(5, is_test=True)
    ]
    predicted_labels = []
    actual_labels = []
    for i in range(len(test_examples)):
        if (i + 1) % 4 == 0:
            sleep(10)
        val = gpt4_query(examples, test_examples[i][0])
        projected_label = label_to_num.get(val)
        actual_label = label_to_num.get(test_examples[i][1]["content"])

        predicted_labels.append(projected_label)
        actual_labels.append(actual_label)
        print(f"Projected label : {projected_label}")
        print(f"Actual label : {actual_label}")
    return {
        "f1-scores": f1_score(
            torch.tensor(predicted_labels), torch.tensor(actual_labels)
        ).tolist(),
        "micro-f1-score": micro_f1_score(
            torch.tensor(predicted_labels), torch.tensor(actual_labels)
        ).item(),
        "macro-f1-score": macro_f1_score(
            torch.tensor(predicted_labels), torch.tensor(actual_labels)
        ).item(),
        "accuracy": accuracy(
            torch.tensor(predicted_labels), torch.tensor(actual_labels)
        ).item(),
    }


def main():
    metrics_dict = {}
    n_shots = 10
    print(f"using approx {n_shots * 700 * 5 * 25} tokens")
    metrics = do_gpt4_evaluation(n_shots)
    print(metrics)
    metrics_dict[str(n_shots)] = metrics
    with open(f"gpt4_results__nshots{n_shots}.json", "w") as json_file:
        json.dump(metrics_dict, json_file)


if __name__ == "__main__":
    main()
