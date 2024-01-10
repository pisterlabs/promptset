import argparse
import asyncio
import json
import os
from asyncio import Semaphore

from datasets import load_dataset
from openai import AsyncOpenAI
import tqdm
from tqdm.asyncio import tqdm_asyncio

PROMPT_TEMPLATE = """
You are an excellent and creative linguist who can rephrase all sentences in many different ways.
Your task is to rephrase the following sentence using different words while sticking as close to the original meaning as possible. Be creative!
You must output a valid JSON object with a single key "sentence" whose value is the rephrased sentence.

BEGIN!

{sentence}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="truthful_qa")
    parser.add_argument("--output", type=str, default="data/rephrased.json")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit_samples", type=int, default=None)
    return parser.parse_args()


def load_truthful_qa():
    ds = load_dataset("truthful_qa", "multiple_choice")["validation"]
    samples = [
        (x["question"],) + tuple(x["mc1_targets"]["choices"]) + tuple(x["mc2_targets"]["choices"])
        for x in ds
    ]
    return ds, samples


def dump_truthful_qa(ds, completions):
    new_ds = []
    for x, completion in zip(ds, completions):
        len1 = len(x["mc1_targets"]["choices"])
        new_ds.append({
            "question": completion[0],
            "mc1_targets": {a: b for a, b in zip(list(completion[1 : len1 + 1]), x["mc1_targets"]["labels"])},
            "mc2_targets": {a: b for a, b in zip(list(completion[len1 + 1:]), x["mc2_targets"]["labels"])},
        })
    return new_ds


async def parallel_complete(args, samples, prompt_template, max_retries=10):
    api_key = args.api_key or os.environ["OPENAI_API_KEY"]
    openai = AsyncOpenAI(api_key=api_key, timeout=30, max_retries=max_retries)
    sem = Semaphore(args.num_workers)

    async def complete(sentence):
        prompt = prompt_template.format(sentence=sentence)
        messages = [{"role": "user", "content": prompt}]
        for i in range(max_retries):
            async with sem:
                response = await openai.chat.completions.create(messages=messages, model=args.model)
            try:
                parsed = json.loads(response.choices[0].message.content)
                return parsed["sentence"]
            except (json.decoder.JSONDecodeError, TypeError):
                # if i == max_retries - 1:
                #     print(f"Could not parse response after retrying: `{response.choices[0].message.content}`")
                #     raise
                pass
            except Exception:
                print(f"An unexpected error occurred trying to parse the response: `{response.choices[0].message.content}`")
                raise
        return None

    sample_lens = [len(x) for x in samples]
    flat_samples = [x for y in samples for x in y]
    tasks = [complete(x) for x in flat_samples]
    flat_completed = await tqdm_asyncio.gather(*tasks)
    completed = []
    for l in sample_lens:
        completed.append(flat_completed[:l])
        flat_completed = flat_completed[l:]
    return completed


def main(args):
    if args.dataset == "truthful_qa":
        ds, samples = load_truthful_qa()
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")
    
    if args.limit_samples is not None:
        ds = ds.select(range(args.limit_samples))
        samples = samples[:args.limit_samples]
    
    completions = asyncio.run(parallel_complete(args, samples, PROMPT_TEMPLATE))

    if args.dataset == "truthful_qa":
        new_ds = dump_truthful_qa(ds, completions)
    else:
        raise Exception(f"Dataset {args.dataset} is not supported")

    with open(args.output, "w") as f:
        json.dump(new_ds, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
