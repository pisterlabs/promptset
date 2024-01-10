"""
File for all similarity tasks:

- SimLex
- SimVerb
- RAW-C

"""

import pandas as pd
import numpy as np
import openai
import backoff  # for exponential backoff
import os
from tqdm import tqdm



PROMPTS = {
    'simlex': "On a scale from 0 (not at all similar) to 10 (the same concept), how similar are '{w1}' and '{w2}'?\n\nSimilarity:",
    'simverb': "On a scale from 0 (not at all similar) to 10 (the same concept), how similar are '{w1}' and '{w2}'?\n\nSimilarity:",
    'raw-c': "On a scale from 0 (totally unrelated) to 4 (same meaning), how <b>related</b> are the uses of the word '{w}' across these two sentences?\n\n{s1}\n{s2}\n\nRelatedness:"

}

SYSTEM_PROMPTS = {
    'simlex': "You are a helpful assistant. Your job is to rate the rate the similarity of two different words.",
    'simverb': "You are a helpful assistant. Your job is to rate the rate the similarity of two different words.",
    'raw-c': "You are a helpful assistant. Your job is to rate the relatedness of the same word in different contexts."
}

def openai_auth():
    """Try to authenticate with OpenAI."""
    ## Read in key
    with open('src/models/gpt_key', 'r') as f:
        lines = f.read().split("\n")
    org = lines[0]
    api_key = lines[1]
    openai.organization = org # org
    openai.api_key = api_key # api_key


def open_instructions(task = "simlex"):
    filepath = "data/raw/{task}/instructions.txt".format(task = task)
    with open(filepath, "r") as f:
        instructions = f.read()
    return instructions


def construct_prompt(task, row):
    prompt = PROMPTS[task]
    if task in ["simlex", "simverb"]:
        w1 = row['word1']
        w2 = row['word2']
        return prompt.format(w1 = w1, w2 = w2)
    if task in ["raw-c"]:
        w = row['word']
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        prompt = prompt.format(w = w, s1 = sentence1, s2 = sentence2)
        return prompt


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def pred_tokens(prompt, system_content, n=10, model="gpt-4"):
    """Get response."""
    output = openai.ChatCompletion.create(
        model = model,
        temperature = 0,
        messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
      max_tokens=10,
      top_p=1
        )

    return output# output['choices'][0]['message']['content']


def main(task="simlex", model='gpt-4'):
    """
    Run GPT-3 on stims.
    """
    # Authenticate
    openai_auth()

    # Assemble data path
    data_path = "data/raw/{task1}/{task2}.csv".format(task1 = task, task2 = task)
    print(data_path)

    # Print config for user
    print(f"Task: \t{task}")
    print(f"Model: \t{model}")

    
    # Read in stims
    df_stimuli = pd.read_csv(data_path)
    print(f"No. stimuli: \t{len(df_stimuli)}")
    print(df_stimuli.head(5))

    # Get instructions
    instructions = open_instructions(task = task)

     # Ensure directory exists
    dirpath = "data/processed/{task}".format(task = task)
    os.makedirs(dirpath, exist_ok=True)
    filename = f"{dirpath}/{task}_{model}.csv".format(task = task)
    file_txt = f"{dirpath}/{task}_{model}.txt".format(task = task)


    answers = []
    col_name = "{m}_response".format(m = model)
    df_stimuli = df_stimuli[433 :len(df_stimuli)]
    for index, row in tqdm(df_stimuli.iterrows(), total=df_stimuli.shape[0]):
        
        
        prompt = construct_prompt(task = task, row = row)
        system_prompt = SYSTEM_PROMPTS[task]
        
        instructions_plus_prompt = instructions + "\n\n" + prompt
        response = pred_tokens(instructions_plus_prompt, system_content = system_prompt, n = 3)
        extracted_response = response['choices'][0]['message']['content']

        row[col_name] = extracted_response 

        answers.append(extracted_response)


        with open(file_txt, "a") as f:
            f.write("{s1},{s2},{response}\n".format(s1 = row['sentence1'],
                                        s2 = row['sentence2'],
                                        response = extracted_response))

    # Create dataframe
    col_name = "{m}_response".format(m = model)
    df_stimuli[col_name] = answers 

    # Ensure directory exists
    dirpath = "data/processed/{task}".format(task = task)
    os.makedirs(dirpath, exist_ok=True)

    # Save file
    filename = f"{dirpath}/{task}_{model}.csv".format(task = task)
    df_stimuli.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    from argparse import ArgumentParser 

    parser = ArgumentParser()

    parser.add_argument("--task", type=str, dest="task",
                        default="raw-c")
    parser.add_argument("--m", type=str, dest="model",
                        default="gpt-3.5-turbo")
    
    args = vars(parser.parse_args())
    main(**args)