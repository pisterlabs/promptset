import os
import torch
import asyncio
import math
import json
import sys
import random
import pandas as pd
import argparse
from tqdm import tqdm
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor

sys.path.append("../../")
sys.path.append("../")
import lmql
import lmql.runtime.bopenai as openai

from utils.dc_baseline_prompter import HFPrompter
from utils.hf_baseline_prompter import HFPrompter as LegacyHFPrompter
from utils.openai_baseline_prompter import OpenAIPrompter

def get_data():
    if not os.path.exists("task.json"):
        os.system('wget https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json')
    # compute sha1
    sha1 = os.popen("sha1sum task.json").read().split(" ")[0]
    assert "6c13a491698efd4b26672919ab918d42dd77afc9" == sha1, "The downloaded task.json has a sha1 mismatch with the expected one (expected: {}, got: {}).".format("6c13a491698efd4b26672919ab918d42dd77afc9", sha1)
    
    with open("task.json") as f:
        data = json.load(f)
    return data

# excluded samples, as they are used as few shot examples
excluded_samples = [
    # "Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?",
    # "Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?"
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method to evaluate.")
    parser.add_argument("--model", type=str, help="Model to evaluate.", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--diff", type=str, help="Existing predictions to not run again.", default=None)
    parser.add_argument("--only-sample", type=str, help="Only the sample with this target label.", default=None, dest="only_sample")
    parser.add_argument("--num-samples", dest="num_samples", type=int, help="Limits the number of samples that are evaluated.", default=None)
    parser.add_argument("--workers", type=int, help="Number of workers to use.", default=16)
    parser.add_argument("--tag", type=str, default=None, help="Tag for the results.")
    return parser.parse_args()

args = get_args()

global module
module = None

@dataclass
class SampleResult:
    correct_item: str
    result_item: str
    result_probs: list
    interaction_trace: str
    query: str = None

async def process_lmql(d):
    if d.get("skip", False):
        return None

    a_to_z = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)"]
    options = list(sorted(d["target_scores"].keys()))
    options_list = ", ".join(options)

    # print(f"""QUESTION = "{d["input"]}\"""")
    # print(f"""OPTIONS = "{options_list}\"""")
    # print(module.lmql_code)
    
    result = await module.query(QUESTION=d["input"], OPTIONS_LIST=options_list)

    if result is None or (type(result) is list and len(result) > 0 and result[0] is None):
        result_item = "ERR"
        result_probs = [(x, 0.0) for x in options]
        reasoning = "<failed to decode full promtp>"
    else:
        result_item = result.variables.get("RESULT")
        result_probs = result.variables.get("P(RESULT)")
        reasoning = result.variables.get("REASONING")
    
    # translate multiple choice answers back
    if result_item.startswith("("):
        # print("translating back multiple choice result", result_item, end="")
        idx = a_to_z.index(result_item)
        result_item = options[idx]
    
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
    # print("RESULT IS", [result_item], [correct_item])
    
    return SampleResult(correct_item, result_item, result_probs, result.prompt, module.lmql_code)

async def baseline(d):
    options = ", ".join(sorted(d["target_scores"].keys()))
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]

    idx = random.randint(0, len(d["target_scores"]) - 1)
    random_result = options.split(", ")[idx]
    return SampleResult(correct_item, random_result, None, None)

async def hf_baseline(d, local=False):
    options = sorted(d["target_scores"].keys())
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
    QUESTION = d["input"]
    
    if args.only_sample is not None:
        if correct_item != args.only_sample:
            print("skipping", correct_item)
            return None
        else:
            print("not skipping", correct_item)

    a_to_z = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)"]
    options = list(zip(a_to_z, options))
    options_list = "".join([f"{a} {o}\n" for a,o in options])

    prompt = f"""
Q:  Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
A: Let's think step by step. 
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937.
So the answer is (D).

Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
A: Let's think step by step. 
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. 
So the answer is (B).

Q: {QUESTION}
Options:
{options_list}
A: Let's think step by step.
""".strip() + ""

    values = [a for a,o in options]
    
    if not local:
        hf = HFPrompter(args.model)
    else:
        hf = LegacyHFPrompter(args.model, local=True)
    
    prompt_with_reasoning = await hf.generate(prompt, max_new_tokens=40, stopping_phrases=["So the answer is"], step_size=20)

    res = await hf.cond_logprob(prompt_with_reasoning + "\nSo the answer is ", values)
    result_item = max(res, key=lambda x: x[1])[0]
    probs = [(x[0], math.exp(x[1])) for x in res]
    original_result_item = result_item

    if result_item.startswith("("):
        # print("translating back multiple choice result", result_item, end="")
        idx = a_to_z.index(result_item)
        result_item = options[idx][1]

    return SampleResult(correct_item, result_item, probs, prompt_with_reasoning.strip() + "\nSo the answer is " + original_result_item)

async def openai_baseline(d):
    oai = OpenAIPrompter(args.model)

    options = sorted(d["target_scores"].keys())
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
    
    QUESTION = d["input"]

    a_to_z = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)"]
    options = list(zip(a_to_z, options))
    options_list = "".join([f"{a} {o}\n" for a,o in options])

    prompt = f"""
Q:  Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
A: Let's think step by step. 
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937.
So the answer is (D).

Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
A: Let's think step by step. 
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. 
So the answer is (B).

Q: {QUESTION}
Options:
{options_list}
A: Let's think step by step.
""".strip() + ""

    values = [a for a,o in options]
    
    prompt_with_reasoning = await oai.generate(prompt, max_new_tokens=40, stopping_phrases=["So the answer is"], step_size=20)

    res = await oai.cond_logprob(prompt_with_reasoning + "\nSo the answer is ", values)
    result_item = max(res, key=lambda x: x[1])[0]
    probs = [(x[0], math.exp(x[1])) for x in res]

    if result_item.startswith("("):
        # print("translating back multiple choice result", result_item, end="")
        idx = a_to_z.index(result_item)
        result_item = options[idx][1]

    return SampleResult(correct_item, result_item, probs, prompt_with_reasoning + "\nSo the answer is " + result_item)

async def process_sample_task(d, sem):
    if d["input"] in excluded_samples:
        print("skipping ", d["input"], "Because it's excluded (selected as few-shot sample)")
        return

    async with sem:
        try:
            if d["model"] == 'baseline':
                return await baseline(d)
            if d["model"] == 'oai':
                return await openai_baseline(d)
            elif d["model"] == "hf":
                return await hf_baseline(d)
            elif d["model"] == "hf-local":
                return await hf_baseline(d, local=True)
            elif d["model"].endswith(".lmql"):
                return await process_lmql(d)
            else:
                assert False, "Unknown model {}".format(d["model"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("ERROR for", d, "Skipping...", module.lmql_code)
            return None

def result_file(file):
    model_name_suffix = ""
    model_name_suffix = f"-{args.model}"
    model_name_suffix = model_name_suffix.replace("/", "-")
    
    tag = ""    
    if args.tag is not None:
        tag = f"{args.tag}-"
        
    if args.num_samples is not None:
        model_name_suffix += f"-n{args.num_samples}"

    return f"results/{tag}{file}-{args.method}{model_name_suffix}.txt"

async def run_eval():
    model = args.method
    print(f"Running with {model}")
    data = get_data()

    global module
    if module is None and model.endswith(".lmql"):
        module = lmql.load(model, autoconnect=True, force_model=args.model)
        sem = asyncio.Semaphore(args.workers)
        module.query.output_writer = lmql.silent
    elif "oai" == args.method:
        lmql.autoconnect()
        sem = asyncio.Semaphore(args.workers)
    else:
        lmql.autoconnect()
        # for other baseline do not process in parallel
        sem = asyncio.Semaphore(args.workers)

    results = []

    n = 0
    n_correct = 0.0

    results = []
    data = data["examples"]
    for d in data: d["model"] = model
    
    if args.num_samples is not None:
        data = data[:args.num_samples]

    if args.diff is not None:
        processed_correct_labels = set()
        with open(args.diff) as f:
            for line in f:
                label = line.split("\t", 1)[0]
                processed_correct_labels.add(label)
        for d in data: 
            correct_label = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
            # print(correct_label, processed_correct_labels)
            d["skip"] = correct_label in processed_correct_labels

    served_model = lmql.model_registry.get(args.model).served_model
    if hasattr(lmql.model_registry.get(args.model), "hf_stats"):
        served_model = lmql.model_registry.get(args.model).hf_stats
    served_model.reset_stats()

    pbar = tqdm(asyncio.as_completed([process_sample_task(d, sem) for d in data]), total=len(data), leave=False)
    for result in pbar:
        result: SampleResult = await result
        n += 1
        
        if result is None: continue
        results += [result]

        result_item = result.result_item
        correct_item = result.correct_item
        n_correct += 1 if (result_item == correct_item) else 0
        
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(result_file("results.csv"), index=False)

        pbar.set_description("N: {}, Accuracy: {:.2f}, Queries: {}, Tokens/Prompt: {}, generate() calls: {}".format(n, n_correct / max(1,n), served_model.num_queries, served_model.billable_tokens / max(1, n), served_model.num_generate_calls))

        if hasattr(served_model, "cost_estimate"):
            pbar.set_description("Accuracy: {:.2f}, Queries: {}, Tokens/Prompt: {}, Cost: ${:.4f}".format(n_correct / max(1,n), served_model.num_queries, served_model.billable_tokens / max(1, n), served_model.cost_estimate(args.model)))

        with open(result_file("score"), "w") as f:
            f.write("Accuracy: {:.2f}\n".format(n_correct / max(1,n)))
            f.write("Queries: {}\n".format(served_model.num_queries))
            f.write("Tokens/Prompt: {}\n".format(served_model.consumed_tokens))
            f.write("generate() calls: {}\n".format(served_model.num_generate_calls))
            f.write("billable tokens: {}\n".format(served_model.billable_tokens))

if __name__ == "__main__":
    asyncio.run(run_eval())
