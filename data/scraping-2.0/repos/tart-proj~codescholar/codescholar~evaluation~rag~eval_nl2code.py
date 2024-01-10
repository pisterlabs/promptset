"""This script is used to evaluate CodeScholar as a retriever for a RAG model.
Here we evaluate CodeScholar + X, where X is a LLM on the NL2Code task."""

import os
import json
import argparse
from typing import List, Dict
from tqdm import tqdm
import time

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from codescholar.evaluation.rag.templates import GPT_FIND_API
from codescholar.evaluation.rag.utils import select_fewshot_examples
from codescholar.evaluation.rag.verify import get_valid_solutions, wrap_check
from codescholar.evaluation.rag.prompt import create_baseline_prompt, create_apidisc_prompt, create_apischolar_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=60))
def gpt_get_predictions(model, prompt, sample, index, verbose) -> List[str]:
    prompts = [
        {"role": "system", "content": "You are a helpful programming assistant who can complete python code given the intent."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=args.n,
        # stop=['"""', "```"],
    )

    predictions = [choice["message"]["content"] for choice in response["choices"]]

    return predictions


def get_prompt(exp, sample, examples, num_tests, function_name):
    if exp == "baseline":
        return create_baseline_prompt(sample, examples, num_tests, function_name)

    elif exp == "apidisc":
        return create_apidisc_prompt(sample, examples, num_tests, function_name)

    elif exp == "apischolar":
        return create_apischolar_prompt(sample, examples, num_tests, function_name)


def eval_gpt(dataset):
    predset = []
    scores_dict = {f"pass@{idx}": [] for idx in range(1, args.n + 1)}

    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="[eval]"):
        examples = select_fewshot_examples(
            sample=sample,
            candidates=dataset[:i] + dataset[i + 1 :],
            num_examples=args.k_shot,
            method=args.fewshot_method,
        )

        prompt = get_prompt(args.experiment, sample, examples, args.num_tests, args.function_name)

        predictions = gpt_get_predictions(
            model=args.model,
            prompt=prompt,
            sample=sample,
            index=i,
            verbose=args.verbose,
        )

        # simple cleansing of predictions
        valid_predictions = get_valid_solutions(predictions, deduplicate=False)
        num_valid = len(valid_predictions)

        assert num_valid == args.n, f"Number of valid predictions {num_valid} != {args.n}"

        scores, outputs = wrap_check(
            sample,
            valid_predictions,
            k=[i + 1 for i in range(num_valid)],
            num_workers=args.n,
            max_num_tests=args.num_tests_eval,
            verbose=args.verbose,
            function_name=args.function_name,
        )

        for idx in range(num_valid):
            key = f"pass@{idx+1}"
            if key in scores:
                scores_dict[key].append(scores[key])

        for output in outputs:
            output[1]["task_id"] = sample["task_id"]

        predset.append(
            {
                "output": outputs,
                "predictions": valid_predictions,
            }
        )

    # write records to prediction file
    json.dump(predset, open(args.output_path, "w"))

    for idx in range(args.n):
        key = f"pass@{idx+1}"
        scores = scores_dict[key]
        print(f"[{key}] {sum(scores)/len(scores):.3f} ({len(scores)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data settings
    parser.add_argument("--input_path", type=str, default="./dataset/cs_rag.jsonl")
    parser.add_argument("--output_path", type=str, required=False)

    # gpt settings
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo"])
    parser.add_argument("--n", type=int, default=10, help="Number of predictions required for each api call.")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1)

    # code-eval settings
    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)

    # function name settings
    parser.add_argument(
        "--function_name", type=str, default="id", choices=["id", "constant", "intent"], help="Method to construct the function name. "
    )

    # fewshot settings
    parser.add_argument("--k_shot", type=int, default=0, help="Number of examples included in the current prompt input. ")
    parser.add_argument(
        "--fewshot_method", type=str, default="random", choices=["random"], help="Method to select the prefix examples for prompt creation."
    )

    # experiment settings
    parser.add_argument("--experiment", type=str, default="baseline", choices=["baseline", "apidisc", "apischolar"])
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.output_path:
        args.output_path = (
            f"res{args.model}-{args.experiment}-n{args.n}-t{args.temperature}-p{args.top_p}-m{args.max_tokens}-k{args.k_shot}.json"
        )

    with open(args.input_path, "r") as fr:
        dataset = [json.loads(l.strip()) for l in fr]

    eval_gpt(dataset)
