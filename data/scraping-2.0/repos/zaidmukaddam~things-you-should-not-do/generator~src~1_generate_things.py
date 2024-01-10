import os
import openai
import json
import argparse
import random
from helpers.helpers import load_samples, load_unused, random_line

openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", help="path to data folder")
parser.add_argument("-n", type=int, default=10)

# Parse and print the results
args = parser.parse_args()
data_folder = args.data_folder
if data_folder is None:
    print("Uh oh!")
    exit(1)

adjectives_filepath = f"{data_folder}/prompt-adjectives.txt"
n = args.n

# BEGIN LOGIC -----------------------------------------------------------------

unused = load_unused(data_folder)
samples = load_samples(data_folder)


def generate_prompt_examples_sample(n: int):
    return random.sample(samples, n)


def generate_prompt(prompt_examples_sample):
    with open(adjectives_filepath, "r") as adjectives_file:
        adjective_1 = random_line(adjectives_file).strip()

    prompt_start = f"Physically Improbable, Convoluted, Preposterous Things You Should Not Do (in the imperative tense):\n"

    prompt = prompt_start
    for sample_example in prompt_examples_sample:
        prompt += f"\n- {sample_example}"
    prompt += "\n-"

    return prompt


def generate(prompt) -> str:
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=1,
        max_tokens=500,
        top_p=1,
        best_of=1,
        frequency_penalty=0.1,  # Too high and we'll have opposite effect - remember the ratio of samples to insertions
        presence_penalty=0.2,
        stop=["\n"],
    )

    text = response.get("choices")[0]["text"]

    return text.strip()


def generate_n(n: int):
    things = []
    for i in range(0, n):
        # Prepend things we've already generated so that penalties apply
        # Prepend because last thing in list has outsized influence
        samples = things + generate_prompt_examples_sample(n)
        prompt = generate_prompt(samples)
        thing = generate(prompt)
        things.append(thing)

    return things


unused += generate_n(n)

with open(f"{data_folder}/unused.json", "w") as file:
    file_things = json.dump(unused, file)
