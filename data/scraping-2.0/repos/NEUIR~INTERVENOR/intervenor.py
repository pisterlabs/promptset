import json
from tqdm import tqdm
import openai
import backoff
import os
import numpy as np
from typing import Iterable, Dict
import gzip
import json
import os
import argparse
from agents.gpt.gpt_wrapper import GPTAgent

GEN_STOP_WORDS = {
    "python":['def'],

}


COR_STOP_WORDS = {
    "python":['def'],
}



def read_problems(problem_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(problem_file)}

def read_results(result_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(result_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def print_options(args,parser):
    message = 'Arguments:\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default_value = parser.get_default(k)
        if v != default_value:
            comment = f'\t(default: {default_value})'
        message += f'{str(k):>30}: {str(v):<40}{comment}\n'

    print(message)


def code_generation(problems,args):
    generator = GPTAgent(args.model)
    stop = GEN_STOP_WORDS[args.language]
    samples = []
    for task_id in tqdm(problems, desc="Generating code", total=len(problems)):
        for _ in range(args.num_samples_per_task):
            code_signature = problems[task_id]["prompt"]
            # print("Task_id: ",task_id)
            # print(code_signature)
            response, usage = generator(code_signature, args.max_tokens, args.temperature, stop)
            # print("--------------------response-------------------")
            # print(response)
            sample = {"task_id": task_id, "completion": response}
            samples.append(sample)

    save_path = os.path.join(".", "results","code_generation",args.dataset, f"{args.language}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Write the code generation results to file :\n",save_path)
    write_jsonl(save_path, samples)




def get_cor_prompt(buggy_code,error_message):
    with open('./prompt/code_teacher.txt', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("%%%buggy_code%%%",buggy_code)
    prompt = prompt.replace("%%%error_message%%%",error_message)
    return prompt

def cor_generation(problems,results,args):
    generator = GPTAgent(args.model)
    stop = COR_STOP_WORDS[args.language]
    samples = []
    for task_id in tqdm(problems, desc="Generate Chain-of-Repairing(CoR)", total=len(problems)):
        for _ in range(args.num_samples_per_task):
            if results[task_id]["result"] == "passed":
                sample = {"task_id": task_id, "completion": results[task_id]["completion"],"result":results[task_id]["result"],"passed":results[task_id]["passed"],"method": ""}
                samples.append(sample)
            else:
                code_signature = problems[task_id]["prompt"]
                buggy_code = code_signature + results[task_id]["completion"] + "\n"
                error_message = results[task_id]["result"]
                prompt = get_cor_prompt(buggy_code,error_message)
                # print(prompt)
                response, usage = generator(prompt, args.max_tokens, args.temperature, stop)
                # print("--------------------response-------------------")
                # print(response)
                sample = {"task_id": task_id, "completion": results[task_id]["completion"],"result":results[task_id]["result"],"passed":results[task_id]["passed"], "method": response}
                samples.append(sample)

    save_path = os.path.join(".", "results", "cor_generation", args.dataset, f"{args.language}_cor.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Write the Chain-of-Repairing(CoR) to file :\n", save_path)
    write_jsonl(save_path, samples)

def get_repair_prompt(buggy_code,error_message,repair_method,code_signature):
    with open('./prompt/code_learner.txt', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("%%%buggy_code%%%", buggy_code)
    prompt = prompt.replace("%%%error_message%%%", error_message)
    prompt = prompt.replace("%%%repair_method%%%", repair_method)
    prompt = prompt +"\n\n"+code_signature
    return prompt
def code_reapiring(problems,results,args):
    generator = GPTAgent(args.model)
    stop = GEN_STOP_WORDS[args.language]
    samples = []
    for task_id in tqdm(problems, desc="Code Repairing", total=len(problems)):
        for _ in range(args.num_samples_per_task):
            if results[task_id]["result"] == "passed":
                sample = {"task_id": task_id, "completion": results[task_id]["completion"],
                          "result": results[task_id]["result"], "passed": results[task_id]["passed"], "method": ""}
                samples.append(sample)
            else:
                code_signature = problems[task_id]["prompt"]
                buggy_code = code_signature + results[task_id]["completion"] + "\n"
                error_message = results[task_id]["result"]
                repair_method = results[task_id]["method"]
                prompt = get_repair_prompt(buggy_code,error_message,repair_method,code_signature)
                # print(prompt)
                response, usage = generator(code_signature, args.max_tokens, args.temperature, stop)
                # print("--------------------response-------------------")
                # print(response)
                sample = {"task_id": task_id, "completion": response,
                          "result": results[task_id]["result"], "passed": results[task_id]["passed"],
                          "method": results[task_id]["method"]}
                samples.append(sample)

    save_path = os.path.join(".", "results", "code_repair", args.dataset, f"{args.language}.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("Write the code repair results to file :\n", save_path)
    write_jsonl(save_path, samples)


def main():
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--problem_file",
        type=str,
        required=True,
        default="",
        help="The file containing the problems.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "humanevalx", "codeerror"],
        help="The dataset to use.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        choices=["python", "cpp", "java", "js"],
        help="The language to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-instruct-0914",
        help="The model to use.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default="512",
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default="0.2",
        help="The temperature to use.",
    )
    parser.add_argument(
        "--num_samples_per_task",
        type=int,
        default="1",
        help="The number of samples to generate per task.",
    )

    parser.add_argument(
        "--todo",
        type=str,
        choices=["code_generation", "cor_generation", "code_reapir"],
        required=True,
        default=None,
        help="Code generation or CoR generation or code reapiring.",
    )
    args = parser.parse_args()
    print_options(args,parser)
    problems = read_problems(args.problem_file)

    if args.todo == "code_generation":
        print("Start code generation...\n\n")
        code_generation(problems,args)
    elif args.todo == "cor_generation":
        print("Start CoR generation...\n\n")
        result_file = os.path.join(".", "results","code_generation",args.dataset, f"{args.language}.jsonl_results.jsonl")
        results = read_results(result_file)
        cor_generation(problems,results,args)
    elif args.todo == "code_reapir":
        print("Start code reapiring...\n\n")
        result_file = os.path.join(".", "results", "cor_generation", args.dataset, f"{args.language}_cor.jsonl")
        results = read_results(result_file)
        code_reapiring(problems,results,args)
if __name__ == '__main__':
    main()
