#!/usr/bin/env python3

from tqdm import tqdm
from more_itertools import chunked
import openai
import backoff
import argparse
import json

import sys, os
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from dataset_util import load_security_dataset

openai.api_key_path = os.expanduser("~/.openai/api_key")

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="code-davinci-002")
parser.add_argument('-d', '--dataset', type=str, required=True)
parser.add_argument("-n", "--num", type=int, default=1, help="Number of completions to generate per task")
parser.add_argument("-t", "--temperature", type=float, default=0.1)
parser.add_argument("-p", "--top_p", type=float, default=1.0)
parser.add_argument("-o", "--output", type=str, default="samples.jsonl")
parser.add_argument("-b", "--batch_size", type=int, default=8)
args = parser.parse_args()

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    response = openai.Completion.create(**kwargs)
    return response

def generate_completions(prompt, suffix, n=8):
    response = completions_with_backoff(
        engine=args.model,
        prompt=prompt,
        suffix=suffix,
        temperature=args.temperature,
        max_tokens=512,
        top_p=args.top_p,
        # stop=args.stop,
        n=n,
    )
    return [choice.text for choice in response.choices]

problems = load_security_dataset(args.dataset)
remaining = len(problems)*args.num
if os.path.exists(args.output):
    already_done = len(open(args.output).readlines())
    remaining -= already_done
    print(f"Already done {already_done} samples, {remaining} remaining")
else:
    already_done = 0
    print(f"Generating {len(problems)*args.num} samples")

i = 0
num_samples_per_task = args.num
with open(args.output, "a") as f, \
     tqdm(total=remaining) as pbar:
    for scenario_id in problems:
        problem = problems[scenario_id]
        prompt = problem["prompt"]
        suffix = problem["suffix"]
        for chunk in chunked(range(num_samples_per_task), args.batch_size):
            if i + len(chunk) <= already_done:
                i += len(chunk)
                pbar.update(len(chunk))
                continue
            completions = generate_completions(prompt, suffix, n=len(chunk))
            for completion in completions:
                pbar.update(1)
                if i < already_done:
                    i += 1
                    continue
                print(json.dumps({
                    "scenario_id": scenario_id,
                    "completion": completion,
                }), file=f)
                f.flush()
                remaining -= 1
