import json
import os
from pathlib import Path
from story_classification import (
    generate_story_classification_dataset,
    SYSTEM_PROMPT,
    ZERO_SHOT_NO_REASONING_PROMPT,
    FEATURE_TO_PROMPT_QUESTION,
    set_all_seeds
)
import openai
import tinystories
import argparse


def main(args):
    """Create a dataset for fine-tuning, upload it, and then start the fine-tuning job."""
    set_all_seeds(args.seed)

    if args.data_cache_dir:
        # make it a python path
        tinystories.DATA_CACHE_DIR = Path(args.data_cache_dir)

    train_dataset, validation_dataset = generate_story_classification_dataset(args.feature, args.dataset_size)
    train_dataset_jsonl, validation_dataset_jsonl = [], []  # type: ignore
    question = FEATURE_TO_PROMPT_QUESTION[args.feature]
    files = {}
    for name, dataset, jsonl in [
        ("train", train_dataset, train_dataset_jsonl),
        ("val", validation_dataset, validation_dataset_jsonl),
    ]:
        print(f"Creating {name} dataset")
        for i, row in dataset.iterrows():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ZERO_SHOT_NO_REASONING_PROMPT.format(story=row["story"], question=question),
                },
                {"role": "assistant", "content": "Yes" if row["label"] == 1.0 else "No"},
            ]
            jsonl.append({"messages": messages})

        # save jsonl to a file, deleting the old one if it exists
        # Each line must be a dictionary
        # Delete existing file
        file_name = f"data/story_classification/{args.feature}_{args.dataset_size}_{name}.jsonl"
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, "w") as f:
            for line in jsonl:
                f.write(json.dumps(line) + "\n")

        # Upload file to openai
        print(f"Uploading {file_name} to openai")
        files[name] = openai.File.create(
            file=open(file_name, "r"),
            purpose="fine-tune",
        )

    print("Starting fine-tuning job")
    job = openai.FineTuningJob.create(
        model=args.model,
        training_file=files["train"].id,
        validation_file=files["val"].id,
        suffix=f"ts-{args.feature}-{args.dataset_size}",
    )
    print(job)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cache_dir", type=str, default="data", help="Adjust data cache dir")
    parser.add_argument("--feature", type=str, default="twist", help="What feature to use")
    parser.add_argument("--dataset_size", type=int, default=100, help="Number of few shot examples to use")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Which openai model to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
