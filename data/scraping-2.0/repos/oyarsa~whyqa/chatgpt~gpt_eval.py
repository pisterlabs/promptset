#!/usr/bin/env python3
# pyright: basic
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from openai import OpenAI
from tqdm import tqdm


MODEL_COSTS = {
    "gpt-3.5-turbo": (  # in: $0.001 / 1K tokens, out: $0.002 / 1K tokens
        0.000001,
        0.000002,
    ),
    "gpt-4": (0.00003, 0.00006),  # in: $0.03 / 1K tokens, out: $0.06 / 1K tokens
}


def calculate_cost(model: str, response: Any) -> float:
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_input, cost_output = MODEL_COSTS[model]
    return input_tokens * cost_input + output_tokens * cost_output


def run_gpt(
    client: OpenAI, model: str, system_prompt: str, message: str
) -> tuple[str, float]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
    )
    result = response.choices[0].message.content
    cost = calculate_cost(model, response)
    return result or "<empty>", cost


SYSTEM_PROMPTS = {
    "simple": """You are a helpful assistant that can evaluate whether an answer is \
correct given a question.""",
}
USER_PROMPTS = {
    "simple": "Based on the story, question and answer, consider the answer is correct \
for the question. Explain your decision.",
    "instructions": """\
Based on the story, question and answer, consider the answer is correct for the \
question. Explain your decision.

Evaluate the answer according to the following criteria:

1. Read the story, the question and the answer. Check if the answer correctly answers \
the question.
2. Make sure that the answer only contains the necessary information.
3. Assign a score for validity on a scale from 1 to 5, where 1 is the lowest and 5 is \
the highest based on the Evaluation Criteria.
4. The story always provides the reason for the question, but it might be implicit. \
Reason about the text instead of merely extracting spans.

Response format:
Explanation: ...
Score: number from 1 to 5
""",
}


def main(
    file: Path,
    outdir: Path = Path("out"),
    n: int = 10,
    rand: bool = True,
    key_file: Path = Path("key"),
    model: str = "gpt-4",
    system_prompt: str = "simple",
    user_prompt: str = "instructions",
    print_messages: bool = True,
) -> None:
    """
    Run a GPT model on the given data and evaluate the results.

    \b
    - file: Path to the json file containing the data (list of objects with keys
        'input', 'output', 'gold', 'valid')
    - n: Number of examples to run
    - rand: Whether to shuffle the data before selecting n examples
    - key_file: Path to the file containing the OpenAI API key (simple text file
        containing only the key)
    - model: Which GPT model to use (gpt-3.5-turbo or gpt-4)
    - system_prompt: Which system prompt to use (only 'simple' for now)
    - user_prompt: Which user prompt to use ('simple' or 'instructions')
    - print_messages: Whether to print the prompt, context, gold and prediction. If
        false, only the progress bar and evaluation results are printed.
    """

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {list(MODEL_COSTS.keys())}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {SYSTEM_PROMPTS.keys()}")
    if user_prompt not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {USER_PROMPTS.keys()}")

    api_key = key_file.read_text().strip()

    data = json.loads(file.read_text())
    if rand:
        random.shuffle(data)

    n = n or len(data)
    if n % 2 != 0:
        n += 1

    # valids = [item for item in data if item["valid"]][: n // 2]
    # invalids = [item for item in data if not item["valid"]][: n // 2]
    # sampled_data = valids + invalids
    sampled_data = data[:n]

    messages: list[tuple[str, str, bool]] = []
    for item in sampled_data:
        story = f'Story: {item["narrative"]}'
        question = f'Question: {item["question"]}'
        # TODO: Actually create and use a prediction
        pred = f'Answer: {item["pred"]}'
        gold = f'Gold: {item["answer"]}'
        valid = True

        display_msg = "\n\n".join([story, question, pred, gold]).strip()
        gpt_msg = "\n\n".join(
            [USER_PROMPTS[user_prompt], story, question, pred]
        ).strip()
        messages.append((display_msg, gpt_msg, valid))

    client = OpenAI(api_key=api_key)
    results: defaultdict[tuple[bool, int], int] = defaultdict(int)
    total_cost = 0

    for display_msg, gpt_msg, valid in tqdm(messages):
        result_s, cost = run_gpt(client, model, SYSTEM_PROMPTS[system_prompt], gpt_msg)
        total_cost += cost

        last_line = result_s.splitlines()[-1].replace("Score:", "").strip()
        result = int(last_line) if last_line.isdigit() else 0
        results[(valid, result)] += 1

        if print_messages:
            print(display_msg)
            print(f"\nGPT: '{result_s}'")
            print("-" * 80)
            print()

    outdir.parent.mkdir(exist_ok=True, parents=True)
    with (outdir / "cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{total_cost}\n")

    df = pd.DataFrame(list(results.items()), columns=["Combination", "Count"])
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    print(df)
    print(f"\nTotal cost: ${total_cost}")


if __name__ == "__main__":
    typer.run(main)
