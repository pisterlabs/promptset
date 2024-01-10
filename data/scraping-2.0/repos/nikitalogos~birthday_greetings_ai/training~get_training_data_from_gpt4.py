import argparse
import json
import os
import traceback

import openai
import yaml
from generate_prompt import get_random_prompt
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


with open(f"{project_root}/auth/.openai.yaml") as f:
    data = yaml.safe_load(f)
    openai.organization = data["organization"]
    openai.api_key = data["api_key"]


def ask_chatgpt(prompt) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    result = completion.choices[0].message.content
    return result


def write_jsonl(prompt: str, completion: str, filename: str) -> None:
    with open(filename, "a") as f:
        json_obj = {"prompt": prompt, "completion": completion}
        json_line = json.dumps(json_obj) + "\n"
        f.write(json_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="Number of training data to generate")
    args = parser.parse_args()

    for i in tqdm(range(args.num), desc="Generating training data"):
        prompt = get_random_prompt()
        print(prompt)
        print("\n~~~~\n")

        try:
            completion = ask_chatgpt(prompt)
        except Exception:
            traceback.print_exc()
            continue
        print(completion)

        write_jsonl(prompt, completion, "training_data.jsonl")
