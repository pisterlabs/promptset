from openai import OpenAI
from decouple import config
import pandas as pd
import argparse
import re
import inspect
from typing import Dict
import csv

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

regex = re.compile(r"\d+\.\s")

def setup():
    client = OpenAI(
        api_key=config("SECRET_KEY")
    )
    
    with open(config("PROMPT_LOC"), 'r') as f:
        initial_prompt = f.read()
    return client, initial_prompt

@retry(wait=wait_random_exponential(min=5, max=60), stop=stop_after_attempt(10))
def run_chatgpt(client: OpenAI, model: str, prompt:str, temp: int):
    """
    Run the ChatGPT model on the input prompt.
    """
    # Define the parameters for the text generation
    completions = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        n=1,
        temperature=temp
    )

    gen_prompt = completions.choices[0].message.content
    return gen_prompt


def generate_issue_prompts(csv_file: str, no_of_samples: int):
    data = pd.read_csv(csv_file)

    # sample issues only when necessary
    if no_of_samples == 0:
        samples = data.copy()
    else:
        samples = data.sample(no_of_samples)

    samples.columns = map(str.lower, samples.columns)

    prompts = dict()
    for _, row in samples.iterrows():
        tags = str(row["tags"])
        title = str(row["issue title"])
        body = inspect.cleandoc(str(row["issue body"]))
        issue_no = str(row["issue number"])
        prompt = f'Q:\nIssue Title: """{title}"""\nIssue Tags: """{tags}"""\nIssue Body:\n"""\n{body}\n"""'

        prompts[issue_no] = re.escape(inspect.cleandoc(prompt))

    return prompts


def generate_full_prompt(initial_prompt: str, issue_prompt: str):
    initial_prompt = initial_prompt.strip()
    issue_prompt = issue_prompt.strip()

    return initial_prompt + "\n\nNow, consider the following " + issue_prompt + "\n\nWhat is the A? Just answer True or False\n"


def classify_answer(answer: str):
    if "true" in answer.lower():
        return "true"
    else:
        return "false"


def main():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    client, initial_prompt = setup()
    parser.add_argument("csv_files", nargs="+", help="Locations of the csv files of issues.")
    parser.add_argument("-n", "--no_of_samples", type=int, default=0, metavar="", help="Number of samples from each csv file. Default will contain all data in csv (number 0)")
    parser.add_argument("-o", "--output_file", metavar="", default="output.csv", help="Output csv file path")
    args = parser.parse_args()

    issue_prompts: Dict[str, str] = dict()
    for csv_file in args.csv_files:
        print("Generating prompt for file: " + csv_file)
        issue_prompts |= generate_issue_prompts(csv_file, args.no_of_samples)

    gpt_name = "gpt-3.5-turbo-1106"
    output_data = list()
    try:
        for issue_no, issue_prompt in tqdm(issue_prompts.items()):
            full_prompt = generate_full_prompt(initial_prompt, issue_prompt)
            answer = run_chatgpt(
                model=gpt_name,
                temp=0,
                prompt=full_prompt,
                client=client
            )
            
            output_data.append({
                "Issue Number": issue_no,
                "Is Bug": classify_answer(answer)
            })
    except:
        # if crashes, pass
        pass

    with open(args.output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["Issue Number", "Is Bug"])
        writer.writeheader()
        writer.writerows(output_data)

    print("output complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        exit(0)
    except FileNotFoundError as e:
        print("Unable to locate file: ", str(e.filename))
        print("Check the file specified in argument, and .env file")
        exit(e.errno)