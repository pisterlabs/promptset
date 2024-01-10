import json
import random
from typing import List, Union, Tuple

import openai

from settings import OPENAI_API_KEY

def get_gpt_logprobs(input: str, model: str) -> Tuple[float, int]:
    res = openai.Completion.create(
        model=model,
        prompt=input,
        max_tokens=0,
        temperature=0,
        logprobs=0,
        echo=True,
        stop=[""],
    )
    res_dict = res["choices"][0]

    maybe_logprobs: List[Union[float, None]] = res_dict["logprobs"]["token_logprobs"]
    logprobs: List[float] = [x or 0.0 for x in maybe_logprobs]
    tokens: List[str] = res_dict["logprobs"]["tokens"]
    offsets: List[int] = res_dict["logprobs"]["text_offset"]

    prompt_offset = len(input)
    completion_tokens = 0
    completion_logprob = 0.0
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == "===":
            break
        elif tokens[i] != "END":
            completion_tokens += 1
            completion_logprob += logprobs[i]
    return completion_logprob, completion_tokens

def calc_gpt_logprobs(data: List[str], model: str, num_shots: int) -> float:
    running_average_logprob = 0.0
    for i, x in enumerate(data):
        if num_shots == 0:
            text = x
        else:
            if i < len(data) / 2:
                primers = random.sample(data[i+1:], num_shots)
            else:
                primers = random.sample(data[:i], num_shots)
            primers.append(x)
            text = "\n\n".join(primers)
        logprob, tokens = get_gpt_logprobs(text, model)
        running_average_logprob += logprob / tokens
    return running_average_logprob / len(data)

def create_text(email: str, summary: str) -> str:
    return email.strip() + "\n\n===\n\n" + summary.strip() + "\nEND"

if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    models = [
        ("ada", 4),
        ("text-ada-001", 0),
        ("ada:ft-personal:cs4248-2022-10-15-05-02-44", 0),
        ("babbage", 4),
        ("text-babbage-001", 0),
        ("babbage:ft-personal:cs4248-2022-10-15-06-31-30", 0),
        ("curie", 4),
        ("text-curie-001", 0),
        ("curie:ft-personal:cs4248-2022-10-15-07-01-21", 0),
    ]

    with open("data/mingann.jsonl", "r") as f:
        json_data = list(map(json.loads, f))
    data = [create_text(x["email"], x["summary"]) for x in json_data]

    for m in models:
        model_name, num_shots = m
        logprob = calc_gpt_logprobs(data=data, model=model_name, num_shots=num_shots)
        print(model_name, num_shots)
        print("Average Log Prob:", round(logprob, 6))
