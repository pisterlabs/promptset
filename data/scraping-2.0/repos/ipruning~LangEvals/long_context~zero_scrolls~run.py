import json
import re

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
from tenacity import retry, wait_chain, wait_fixed
from tqdm import tqdm

import tiktoken

API_KEY = None

def num_tokens_from_message(message):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    num_tokens += len(encoding.encode(message))
    return num_tokens

def extract_ans(ans, mode):
    options = [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
        "(I)",
        "(J)",
        "(K)",
        "(L)",
        "(M)",
        "(N)",
        "(O)",
        "(P)",
        "(Q)",
        "(R)",
        "(S)",
        "(T)",
        "(U)",
        "(V)",
        "(W)",
        "(X)",
        "(Y)",
        "(Z)",
    ]

    ans = ans.strip()

    if mode == "multiple_choice":
        for option in options:
            if option in ans:
                return option[1]
    if mode == "free_form_v1":
        lower_ans = ans.lower()
        if lower_ans.startswith("yes"):
            return "Yes"
        elif lower_ans.startswith("no"):
            return "No"
        elif lower_ans.startswith("unanswerable"):
            return "Unanswerable"
        else:
            return ans
    if mode == "free_form_v2":
        return ans[ans.find(":") + 1 :].strip()

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] + [wait_fixed(5) for i in range(2)] + [wait_fixed(10)]))
def completion_with_backoff(model_index, messages):
    endpoint = Anthropic(
        api_key=API_KEY
    )

    completion = endpoint.completions.create(
        model=model_index,
        max_tokens_to_sample=1000,
        prompt=f"{messages}",
    )

    return completion.completion

def qalt():
    BENCHMARK = "long_context/zero_scrolls"
    TASK = "qalt"
    MODE = "test"

    with open(f"{BENCHMARK}/{TASK}/{MODE}.jsonl", "r") as f, open(
        f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "w"
    ) as out_f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            q = data["input"]
            prompt = f"""
            {HUMAN_PROMPT} You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D).
            Choose the best answer by writing its corresponding letter (either A, B, C, or D).
            I will ask you multiple choice questions and you will respond to me in a format like (A) (B) (C) (D).
            {AI_PROMPT} I understand, and I'll answer you in a format like (A) (B) (C) (D) and give my answer at the end of the respond.
            {HUMAN_PROMPT} {q} {AI_PROMPT}"""

            response = completion_with_backoff("claude-2.0", prompt)
            model_ans = extract_ans(response, "multiple_choice")

            print("\n")
            print(response)
            print("\n")
            print(model_ans)
            print("\n")

            data["model_prompt"] = prompt
            data["model_response"] = response
            data["output"] = model_ans

            out_f.write(json.dumps(data) + "\n")

    result_dict = {}

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "r") as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data["id"]
            output = data["output"]
            result_dict[example_id] = output

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred_out.json", "w") as out_f:
        json.dump(result_dict, out_f, indent=4)

def qspr():
    BENCHMARK = "long_context/zero_scrolls"
    TASK = "qspr"
    MODE = "test"

    with open(f"{BENCHMARK}/{TASK}/{MODE}.jsonl", "r") as f, open(
        f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "w"
    ) as out_f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            q = data["input"]
            prompt = f"""
            {HUMAN_PROMPT} You are given a scientific article and a question.
            Answer the question as concisely as you can, using a single phrase or sentence if possible.
            If the question cannot be answered based on the information in the article, write \"unanswerable\".
            If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\"
            {AI_PROMPT} I understand, I will answer the question as concisely as I can, using a single phrase or sentence if possible. I will start my answer with \"The answer is :\".
            {HUMAN_PROMPT} {q} {AI_PROMPT} The answer is:"""

            response = completion_with_backoff("claude-2.0", prompt)
            model_ans = extract_ans(response, "free_form_v1")

            print("\n")
            print(response)
            print("\n")
            print(model_ans)
            print("\n")

            data["model_prompt"] = prompt
            data["model_response"] = response
            data["output"] = model_ans

            out_f.write(json.dumps(data) + "\n")

    result_dict = {}

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "r") as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data["id"]
            output = data["output"]
            result_dict[example_id] = output

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred_out.json", "w") as out_f:
        json.dump(result_dict, out_f, indent=4)

def bkss():
    BENCHMARK = "long_context/zero_scrolls"
    TASK = "bkss"
    MODE = "test"

    with open(f"{BENCHMARK}/{TASK}/{MODE}.jsonl", "r") as f, open(
        f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "w"
    ) as out_f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            q = data["input"]
            prompt = f"""
            {HUMAN_PROMPT} {q} {AI_PROMPT}"""

            response = completion_with_backoff("claude-2.0", prompt)
            model_ans = extract_ans(response, "free_form_v2")

            print("\n")
            print(response)
            print("\n")
            print(model_ans)
            print("\n")

            data["model_prompt"] = prompt
            data["model_response"] = response
            data["output"] = model_ans

            out_f.write(json.dumps(data) + "\n")

    result_dict = {}

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred.jsonl", "r") as f:
        for line in f:
            data = json.loads(line.strip())
            example_id = data["id"]
            output = data["output"]
            result_dict[example_id] = output

    with open(f"{BENCHMARK}/{TASK}/{MODE}_pred_out.json", "w") as out_f:
        json.dump(result_dict, out_f, indent=4)


qalt()
qspr()
bkss()
