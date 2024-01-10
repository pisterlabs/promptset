from etk.data.mathqa_dataset import read_gsm8k
from etk.eval_utils import batch_loader
from etk.execution import check_correctness
from tqdm import tqdm
import re
import random
import json
import sys
import openai
from ratelimit import limits, sleep_and_retry
from itertools import zip_longest
import time


@sleep_and_retry
@limits(calls=1, period=60)
def call_api(engine, prompt, max_tokens, n, temperature):
    return openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        temperature=temperature,
    )


def gsm8k_test_of_float(answer):
    return f"assert (abs(answer - {answer})/max(abs({answer}), 1e-5))<0.01"


def gsm8k_prompt(few_shot_prompt, text):
    return few_shot_prompt + text


def gsm8k_program_of_completion(completion):
    re_key = "\nanswer.*?\n"
    search_result = re.search(re_key, completion)

    if search_result:
        return completion[:search_result.span()[1]]
    else:
        return completion


def main():
    random.seed(20)

    cfg_path = sys.argv[1]
    with open(cfg_path) as f: 
        cfg = yaml.safe_load(f)

    num_prompts = cfg["batch_size"]
    n = cfg["k"]
    temp = cfg["temp"]
    filename = cfg["out_file"]
    split = cfg["split"]
    dataset = cfg["dataset"]

    prompt = open("gsm8k_prompt.txt", "r").read()

    if dataset=="gsm8k": 
        train_data = read_gsm8k(f"../data/gsm8k/gsm8k_{split}.jsonl")
    else: 
        raise KeyError("dataset field in config invalid")

    dataloader = batch_loader(train_data, num_prompts)

    for batch in tqdm(dataloader):
        if dataset == "gsm8k":
            tests = [gsm8k_test_of_float(instance.answer) for instance in batch]
            prompts = [gsm8k_prompt(prompt, instance.text) for instance in batch]
            headers = ["" for _ in batch]
        else: 
            raise KeyError("dataset field in config invalid")

        nls = [instance.text for instance in batch]
        task_ids = [instance.task_id for instance in batch]

        outputs = call_api(
            engine="code-davinci-002",
            prompt=prompts,
            max_tokens=150,
            n=n,
            temperature=temp,
        )

        outputs = [output["text"] for output in outputs["choices"]]

        for out, task_id, nl, header, test in zip(
            batch_loader(outputs, n), task_ids, nls, headers, tests
        ):

            if dataset == "gsm8k":
                bodies = [gsm8k_program_of_completion(completion) for completion in out]
            else: 
                raise KeyError("dataset field in config invalid")

            problem = {"prompt": header, "test": test}

            passed_lst = [
                check_correctness(problem, x, timeout=1)["passed"] for x in bodies
            ]

            if True in passed_lst:
                gold_code = bodies[passed_lst.index(True)]
                passed = 1
            else:
                gold_code = False
                passed = 0

            pass_1 = sum(passed_lst) / len(passed_lst)

            with open(filename + ".jsonl", "a+") as f:
                record = json.dumps(
                    {
                        "task_id": task_id,
                        "text": nl,
                        "test": test,
                        "gold_solution": gold_code,
                        "passk": passed,
                        "pass1": pass_1,
                        "passed_lst": passed_lst,
                        "samples": bodies,
                    }
                )
                f.write(record + "\n")

    with open(filename + ".jsonl") as f:
        log = [json.loads(line) for line in f]

    num_examples = len(log)

    num_passed = sum([x["passk"] for x in log])
    pass_k = num_passed / num_examples

    pass_1 = sum([x["pass1"] for x in log]) / num_examples

    to_dump = {"passk": pass_k, "pass1": pass_1, "log": log}

    with open("results/" + filename + ".json", "w") as fle:
        json.dump(to_dump, fle)

if __name__=="__main__": 
    main()
