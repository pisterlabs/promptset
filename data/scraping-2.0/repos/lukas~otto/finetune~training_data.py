import re
import argparse
from openai import ChatCompletion
import json
import os

import openai

openai.api_base = "https://api.wandb.ai/proxy/openai/v1"
wandb_key = os.environ["WANDB_API_KEY"]
openai_key = os.environ['OPENAI_API_KEY']
openai.api_key = f"{wandb_key}:{openai_key}"


def parse_example_file(file) -> (str, list[dict], list[dict]):
    with open(file, "r") as f:
        cur_user_prompt = ""
        cur_answer = ""
        section = ""
        line_number = 0
        prompt = ""
        examples = []
        prompt_examples = []

        for line in f:
            if line.startswith("####"):
                # section header
                if line.startswith("#### Prompt:"):
                    section = "prompt"
                elif line.startswith("#### Prompt Examples:"):
                    section = "prompt examples"
                elif line.startswith("#### Examples:"):
                    section = "examples"
                else:
                    raise ValueError(
                        f"Malformed section header in line {line_number} in {file}"
                    )
            else:
                if section == "prompt":
                    if line.strip != "":
                        prompt += line
                elif section == "examples" or section == "prompt examples":
                    if line.startswith("### User:"):
                        if cur_user_prompt != "":
                            raise ValueError(
                                f"Malformed file: two user prompts in a row in line {line_number} in {file}"
                            )
                        # get the string after ### User:
                        cur_user_prompt = line.split(":", 2)[1].strip()

                    elif line.startswith("### Assistant:"):
                        # get part of line after the :
                        cur_answer = line.split(":", 1)[1].strip()
                        example = {"user": cur_user_prompt,
                                   "answer": cur_answer}
                        if section == "prompt examples":
                            prompt_examples.append(example)
                        else:
                            examples.append(example)
                        cur_user_prompt = ""
                    elif line.startswith("###"):
                        raise ValueError(
                            f"Malformed file: line needs to start with ### User or ### Assistant at line {line_number} in {file}"
                        )

            line_number += 1
        return (prompt, prompt_examples, examples)


def create_training_data_file(files):
    examples = []
    for file in files:
        prompt, prompt_examples, file_examples = parse_example_file(file)
        examples.append(file_examples)

    for example in examples:
        for line in example:
            print(json.dumps(line))


def prompt_examples_to_string(prompt_examples: [dict]):
    prompt_examples_str = ""
    for example in prompt_examples:
        prompt_examples_str += (
            f"### User: {example['user']}\n### Assistant: {example['answer']}\n"
        )
    return prompt_examples_str


def create_prompt(prompt: str, prompt_examples: list[dict]):
    data_collection_prompt = f"""
I am collecting training data for a voice assistant.

The voice assistant has the command:
{prompt}
Some examples of how a user might say this command and the response is:
{prompt_examples_to_string(prompt_examples)}
Please give me more examples of ways that a user might query this command and the correct response.

Please make sure the examples start with ### User and ### Assistant.
"""

    return data_collection_prompt


def create_prompts(files: [str]):
    for file in files:
        prompt, prompt_examples, file_examples = parse_example_file(file)

        data_collection_prompt = create_prompt(prompt, prompt_examples)
        print(data_collection_prompt)


def collect_training_data(file: str):
    prompt, prompt_examples, file_examples = parse_example_file(file)
    data_collection_prompt = create_prompt(prompt, prompt_examples)
    print(data_collection_prompt)

    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are helpful assistant generating training data for a voice assistant.",
            },
            {"role": "user", "content": data_collection_prompt},
        ],
    )
    answer = response["choices"][0]["message"]["content"]
    print(answer)


def strip_ansi_codes(s):
    return re.sub(r"\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?", "", s)


def generate_other_examples(files):
    lines = set()  # set of lines - want to avoid duplicates
    for file in files:
        with open(file, "r") as f:
            for line in f:
                line = strip_ansi_codes(line)
                line = line.strip()
                if (
                    line == ""
                    or line.startswith("(")
                    or line.startswith("[")
                    or line.startswith("*")
                    or line.startswith(".")
                    or line.startswith("-")
                ):
                    continue
                else:
                    lines.add(line)

    for line in lines:
        print(f"### User: {line}\n### Assistant: other()\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(

        description='Generate training data from example files')

    argparser.add_argument('--files', metavar='file', type=str, nargs='+',
                           help='example files to parse')
    argparser.add_argument('-t', '--training-data-file', action='store_true',
                           help="generate training data file")
    argparser.add_argument('-p', '--prompt', action='store_true',
                           help="generate training data collection prompt")
    argparser.add_argument('-c', '--collect-training-data', action='store_true',
                           help="collect training data")
    argparser.add_argument('-o', '--generate-other-examples', action='store_true',
                           help="generate other examples")

    args = argparser.parse_args()
    if args.training_data_file:
        create_training_data_file(args.files)
    elif args.prompt:
        create_prompts(args.files)
    elif args.collect_training_data:
        collect_training_data(args.files[0])
    elif args.generate_other_examples:
        generate_other_examples(args.files)
