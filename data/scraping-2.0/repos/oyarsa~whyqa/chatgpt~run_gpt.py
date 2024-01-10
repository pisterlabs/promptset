# pyright: basic
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from openai import OpenAI
from tqdm import tqdm

from gpt_eval import calculate_cost


def process_dataset(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    dataset: list[dict[str, Any]],
    print_messages: bool,
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    total_cost = 0

    for item in tqdm(dataset):
        message = "\n\n".join(
            [
                user_prompt,
                f'Narrative: {item["narrative"]}',
                f'Question: {item["question"]}',
            ]
        )

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
        results.append(
            {
                "narrative": item["narrative"],
                "question": item["question"],
                "answer": item["answer"],
                "pred": result,
                "model_used": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        total_cost += calculate_cost(model, response)

        if print_messages:
            answerable = item["is_ques_answerable_annotator"] == "Answerable"
            print(message)
            print(f"\nAnswerable: {answerable}")
            print(f"\nAnswer: {item['answer']}")
            print(f"\nGPT: '{result}'")
            print()
            print("-" * 80)
            print()

    return results, total_cost


SYSTEM_PROMPTS = {
    "simple": """You are a helpful assistant that can answer questions about why \
things happen."""
}
USER_PROMPTS = {
    "simple": """Based on the story, answer the question. The answer might be implicit \
in the text. The response should be just the answer, nothing else.""",
    "instructions": """\
""",
}


def main(
    file: Path,
    output: Path,
    n: int = 10,
    rand: bool = False,
    key_file: Path = Path("key"),
    model: str = "gpt-3.5-turbo",
    system_prompt: str = "simple",
    user_prompt: str = "simple",
    print_messages: bool = True,
    include_unanswerable: bool = False,
) -> None:
    dataset = json.loads(file.read_text())
    if not include_unanswerable:
        dataset = [
            d for d in dataset if d["is_ques_answerable_annotator"] == "Answerable"
        ]
    if rand:
        random.shuffle(dataset)

    n = n if n > 0 else len(dataset)
    dataset = dataset[:n]

    client = OpenAI(api_key=key_file.read_text().strip())

    processed_data, total_cost = process_dataset(
        client,
        model,
        SYSTEM_PROMPTS[system_prompt],
        USER_PROMPTS[user_prompt],
        dataset,
        print_messages,
    )
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(processed_data, indent=4))
    print("Total cost:", total_cost)

    with (output.parent / "cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{total_cost}\n")


if __name__ == "__main__":
    typer.run(main)
