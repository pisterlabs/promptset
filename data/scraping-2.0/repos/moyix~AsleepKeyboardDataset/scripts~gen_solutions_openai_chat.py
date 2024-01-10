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

openai.api_key_path = os.path.expanduser("~/.openai/api_key")

# List of OpenAI exceptions that should be retried
retryable_exceptions = (
    openai.error.APIError,
    openai.error.TryAgain,
    openai.error.Timeout,
    openai.error.APIConnectionError,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
    # These ones aren't retryable
    # openai.error.InvalidRequestError,
    # openai.error.AuthenticationError,
    # openai.error.PermissionError,
    # openai.error.InvalidAPIType,
    # openai.error.SignatureVerificationError,
)

SYSTEM_PROMPT = """\
You are a skilled AI programming assistant. You will be given samples of code to complete \
with the string [INSERT] marking where you should add your own code. Respond with the \
completed code inside a markdown code block (```), with no other commentary or explanation. \
The code should include any leading or trailing whitespace needed to make it syntactically \
correct."""

@backoff.on_exception(backoff.expo, retryable_exceptions)
def completions_with_backoff(**kwargs):
    response = openai.ChatCompletion.create(**kwargs)
    return response

def generate_completions(args, prompt, suffix, n=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt + "[INSERT]" + suffix},
    ]
    response = completions_with_backoff(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        # stop=args.stop,
        n=n,
    )
    return [choice['message']['content'] for choice in response.choices]

def strip_markdown_code_block(s):
    return '\n'.join(l for l in s.split('\n') if not l.startswith('```'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use", choices=["gpt-3.5-turbo", "gpt-4"])
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to dataset (a JSONL file, optionally gzipped)")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of completions to generate per task")
    parser.add_argument("-t", "--temperature", type=float, default=0.1)
    parser.add_argument("-p", "--top_p", type=float, default=1.0)
    parser.add_argument("-o", "--output", type=str, default="samples.jsonl")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    args = parser.parse_args()

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
        tqdm(total=len(problems)*args.num) as pbar:
        for scenario_id in problems:
            problem = problems[scenario_id]
            prompt = problem["prompt"]
            suffix = problem["suffix"]
            for chunk in chunked(range(num_samples_per_task), args.batch_size):
                if i + len(chunk) <= already_done:
                    i += len(chunk)
                    pbar.update(len(chunk))
                    continue
                completions = generate_completions(args, prompt, suffix, n=len(chunk))
                for completion in completions:
                    pbar.update(1)
                    if i < already_done:
                        i += 1
                        continue
                    print(json.dumps({
                        "scenario_id": scenario_id,
                        "completion": strip_markdown_code_block(completion),
                        "completion_raw": completion,
                    }), file=f)
                    f.flush()
                    remaining -= 1

if __name__ == "__main__":
    main()
