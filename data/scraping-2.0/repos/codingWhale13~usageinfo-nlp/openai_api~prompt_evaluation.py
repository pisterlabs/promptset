#!/usr/bin/env python3
import time
import os
import argparse
import json
import pandas as pd

from evaluation.scoring.metrics import Metrics
from evaluation.scoring.core import gpt_predictions_to_labels
from openai_pre_annotion import pre_label_with_logprobs
from openai_api.openai_smart_pre_annotation import pre_label_with_dynamic_samplling


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="Evaluate various prompts and their OpenAI performance."
    )
    arg_parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="json file path with the golden labelled reviews",
    )
    arg_parser.add_argument(
        "--prompts", "-p", default="prompts.json", help="json file path with prompts"
    )
    arg_parser.add_argument(
        "--selected-prompts",
        "-s",
        default="",
        help="Select prompts name delimited with , from your prompts file",
    )

    return arg_parser.parse_args(), arg_parser.format_help()


def label_with_prompts(reviews, prompt):
    print(f"Labelling with prompt: {prompt['name']}")
    if prompt["name"] not in reviews[0]["label"].keys():
        print(
            f"No existing OpenAI labels for prompt {prompt['name']} found. Starting api querying..."
        )
        for review in reviews:
            review["label"][prompt["name"]] = pre_label_with_logprobs(
                review, prompt=prompt["prompt"]
            )
    else:
        print(
            f"Existing OpenAI labels for prompt {prompt['name']} found. Skipping api querying..."
        )
    return reviews


def main():
    args, help_text = parse_args()

    reviews_file_name = os.path.abspath(args.file)
    prompts_file_name = os.path.abspath(args.prompts)
    use_only_selected_prompt = args.selected_prompts != ""
    selected_prompts = args.selected_prompts.strip().split(",")
    with open(reviews_file_name, "r") as reviews_file, open(
        prompts_file_name, "r+"
    ) as prompts_file:
        review_json = json.load(reviews_file)
        if type(review_json) is list:
            reviews = review_json
        else:
            reviews = review_json["reviews"]

        prompts_json = json.load(prompts_file)

        for prompt in prompts_json:
            if use_only_selected_prompt and prompt["name"] not in selected_prompts:
                print(f"Skipping prompt: {prompt['name']}")
                continue
            label_with_prompts(reviews, prompt)
            print(f"Finished labelling for prompt: {prompt['name']}")

            labels = gpt_predictions_to_labels(reviews, prompt_ids=[prompt["name"]])
            review_scores, prompt["score"] = Metrics(labels).calculate()
            for review_score, review in zip(review_scores, reviews):
                if review_score["review_id"] == review["review_id"]:
                    review["label"][prompt["name"]]["scores"] = review_score["scores"]
        output_file_name = (
            os.path.dirname(reviews_file_name)
            + f"/prompt_labelled_"
            + os.path.basename(reviews_file_name)
        )
        with open(output_file_name, "w") as reviews_file:
            json.dump(review_json, reviews_file)

        prompts_file.truncate(0)
        prompts_file.seek(0)
        json.dump(prompts_json, prompts_file)


if __name__ == "__main__":
    main()
