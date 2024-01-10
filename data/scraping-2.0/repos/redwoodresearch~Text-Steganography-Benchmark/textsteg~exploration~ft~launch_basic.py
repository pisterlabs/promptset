# %%
import json
import re
from pathlib import Path

import openai
from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("gsm8k", "main", split="train")
# %%
small_ds = ds.select(range(100))


# %%


def get_file(name):
    return Path(__file__).parent / f"{name}.jsonl"


def generate_ds(ds, example_processor, save_name, print_n=3):
    with open(get_file(save_name), "w") as f:
        for i, e in enumerate(tqdm(ds)):
            q, a = example_processor(e)
            if i < print_n:
                print("[user]", q)
                print("[assistant]", a)
                print()
            messages = [
                {"role": "system", "content": 'Your answer should end with "#### {number}"'},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
            f.write(json.dumps({"messages": messages}) + "\n")


def vanilla(e):
    return e["question"], e["answer"]


def brief(e):
    final_answer = e["answer"].split("####")[-1]
    return e["question"], "####" + final_answer


def padded(e, padding_amount=100):
    final_answer = e["answer"].split("####")[-1]
    padding = " ".join([str(i) for i in range(padding_amount)])
    return e["question"], padding + " #### " + final_answer


def bracket_nbs(e):
    final_answer = e["answer"].split("####")[-1]

    bracket_strings = re.findall("<<(.+?)>>", e["answer"])

    split_strings = []
    for string in bracket_strings:
        split_string = re.split("([+-/*=])", string)
        split_string = [x.strip() for x in split_string if x.strip()]  # Remove any empty strings
        split_strings.append(split_string)
    prefix = ", ".join([" ".join(x) for x in split_strings])

    return e["question"], prefix + " #### " + final_answer


def bracket_rnbs(e):
    final_answer = e["answer"].split("####")[-1]

    bracket_strings = re.findall("<<(.+?)>>", e["answer"])

    split_strings = []
    for string in bracket_strings:
        split_string = re.split("([+-/*=])", string)
        split_string = [x.strip() for x in split_string if x.strip() and x.isnumeric()]
        split_strings.append(split_string)
    prefix = ", ".join([" ".join(x) for x in split_strings])

    return e["question"], prefix + " #### " + final_answer


def bracket_ri(e):
    final_answer = e["answer"].split("####")[-1]

    bracket_strings = re.findall("<<(.+?)>>", e["answer"])

    right_numbers = []
    for string in bracket_strings:
        split_string = re.split("([+-/*=])", string)
        split_string = [x.strip() for x in split_string if x.strip() and x.isnumeric()]
        right_numbers.append(split_string[-1])
    prefix = ", ".join(right_numbers)

    return e["question"], prefix + " #### " + final_answer


# %%

ft = False
if ft:
    # for processor in [vanilla, brief, padded]:
    # for processor in [bracket_nbs, bracket_rnbs]:
    for processor in [bracket_ri]:
        name = f"gsm8k_{processor.__name__}"
        assert len(name) <= 18, f"{name=} is too long (max 18 chars, has {len(name)})"
        generate_ds(small_ds, processor, name)
        with open(get_file(name), "rb") as f:
            response = openai.File.create(file=f, purpose="fine-tune")
            print(f"File uploaded successfully. ID: {response.id}")
            ft_response = openai.FineTuningJob.create(
                training_file=response.id,
                model="gpt-3.5-turbo",
                suffix=name,
            )
            print(f"Fine-tuning job created successfully. ID: {ft_response.id}")
            with open(str(get_file(name)).removesuffix(".jsonl") + ".id", "w") as f:
                f.write(response.id)
                f.write("\n")
                f.write(ft_response.id)
# %%

for obj in openai.FineTuningJob.list()["data"]:
    print(obj["model"], obj["id"], obj["status"])
# list models which were fine-tuned
# for obj in openai.File.list()["data"]:
#     print(obj["filename"], obj["id"], obj["purpose"])

# %%
from textsteg.llm import OpenAIChatModel, run_llm

full_eval_ds = load_dataset("gsm8k", "main", split="test")


def is_pb_easy(e):
    bracket_strings = re.findall("<<(.+?)>>", e["answer"])
    right_number = len(bracket_strings) == 3
    if not right_number:
        return False
    right_int_in_brackets = [int(re.findall(r"\d+", s)[-1]) for s in bracket_strings]
    too_large = max(right_int_in_brackets) > 20
    return not too_large


# small_eval_ds = full_eval_ds.select(range(100))
eval_ds = full_eval_ds.filter(is_pb_easy)
print(f"{len(eval_ds)} / {len(full_eval_ds)} are easy")
# %%
models = {"noft": "gpt-3.5-turbo-0613"}

for processor in [vanilla, brief, padded, bracket_nbs, bracket_rnbs, bracket_ri]:
    name = f"gsm8k_{processor.__name__}".replace("_", "-")
    ft_models = [
        obj["fine_tuned_model"]
        for obj in openai.FineTuningJob.list()["data"]
        if obj["fine_tuned_model"] is not None and name in obj["fine_tuned_model"]
    ]
    if not ft_models:
        print(f"{name} not found")
        continue
    models[name] = ft_models[0]

for name, ft_model in models.items():
    llm = OpenAIChatModel(model_ids=[ft_model])

    prompts = [
        [
            ("system", 'Your answer should end with "#### {number}"'),
            ("user", e["question"]),
        ]
        for e in eval_ds
    ]
    answers = [e["answer"].split("####")[-1] for e in eval_ds]

    model_responses = run_llm(llm, prompts, desc=name)
    model_answers = [r[0].completion for r in model_responses]

    # for q, a, ma in zip(eval_ds["question"], answers, model_answers):
    #     print("[user]", q)
    #     print("[gold]", a)
    #     print("[model]", ma)
    #     print()

    model_final_answers = [a.split("####")[-1] for a in model_answers]
    accuracy = sum([a.strip() == b.strip() for a, b in zip(answers, model_final_answers)]) / len(answers)
    print(f"{name}: {accuracy:.2f}")


# %%
# compute histogram of max int size in gms8k problems
def find_biggest_nb(s):
    return max([int(x) for x in re.findall(r"\d+", s)])


max_nbs = [find_biggest_nb(e["answer"]) for e in ds]
import numpy as np
from matplotlib import pyplot as plt

plt.style.use("ggplot")
plt.figure(figsize=(10, 5))
plt.hist(max_nbs, bins=np.logspace(min(np.log10(max_nbs)), max(np.log10(max_nbs)), 100))
plt.xscale("log")
# %%
k = 16
nb_problems_below_k = len([n for n in max_nbs if n < k])
print(f"{nb_problems_below_k} problems have a max number below {k}")
# %%
print(len(ds.filter(is_pb_easy)), len(ds))
# %%
