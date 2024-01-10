import os
import base64
import requests
from openai import OpenAI
import time
import argparse
import pandas as pd
from tqdm import tqdm
import json
import csv
import numpy as np

#Model hyper parameters
MODEL = "gpt-4-vision-preview"
TEMPERATURE = 1.4
RANDOM_SEED = 123
NUM_TRIES = 1

#Directories to get images from
DIR_IMAGES = "llm_baseline/img"

#Directories to get input files from
DIR_DATA = "data"

#Directories to save files to
DIR_EXPERIMENTS = "experiments"
DIR_BASELINE = "llm_baseline"
DIR_PROMPTS = "prompts"

PROMPT_PREFIX = """
Look at the 2-D physics scenario in the image. It portrays a set of blocks (purple and red) sitting on a grey platform.

The blocks may be stacked stably or unstably. For a sufficiently long amount of time, gravity is allowed to act on the blocks.

Your job is to answer the following question on a 1-7 scale, with 1 being “Definitely No” and 7 being “Definitely Yes”. 

Only answer "1" or "7" if you are extremely sure about the outcome of the question, as it pertains to the image.

Question as it pertains to image:
"""

FEW_SHOT_EXAMPLES = """
Here are some examples:
"""

PROMPT_SUFFIX = """
Only answer with 1 number, ranging from 1-7 inclusive. Again only answer with just one token.
"""


def construct_prompt(df_query, task_id):

    question = df_query.loc[task_id, 'english_question']
    box_ensemble_index = df_query.loc[task_id, 'box_ensemble_index']

    prompt = PROMPT_PREFIX
    prompt += question
    prompt += PROMPT_SUFFIX

    image_path = os.path.join(DIR_IMAGES, f"ensemble{box_ensemble_index}.png")

    return prompt, image_path

def construct_few_shot_prompt(df_query, df_examples, task_id):

    image_index = (task_id % 6) // 6
    example_task_ids = [image_index + i for i in range(0,6) if (image_index + i)!=task_id]

    question = df_query.loc[task_id, 'english_question']
    box_ensemble_index = df_query.loc[task_id, 'box_ensemble_index']

    prompt = PROMPT_PREFIX
    prompt += question
    prompt += PROMPT_SUFFIX

    image_path = os.path.join(DIR_IMAGES, f"ensemble{box_ensemble_index}.png")

    return prompt, image_path

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_llm(prompt, image_path):

    base64_image = encode_image(image_path)
    
    i = 0
    completion = None
    client = OpenAI()
    while i < NUM_TRIES:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type":"text", "text": prompt},
                            {
                                "type":"image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                    ],
                temperature=TEMPERATURE,
            )
            i = NUM_TRIES #Break the loop if completion sucseeded
        except Exception as e:
            print(e)
            print("WARNING: Failed to querey OpenAI. Trying again")
        i+= 1

    if completion is None:
        raise ValueError("ERROR: Not able to query openai sucessfully")
    
    return completion.choices[0].message.content.strip()

def parse(llm_output):

    try:
        parsed_output = int(llm_output.strip())
    except:
        parsed_output = 4
        print("Couldn't parse LLM output. Giving neutral answer")
    
    return parsed_output

def run_experiment(input_file_name: str, experiment_id: str = None, n_participants: int = 1):
    experiment_id = experiment_id or datetime.datetime.now().strftime('run-%Y-%m-%d-%H-%M-%S')
    ckpt_dir = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_BASELINE)
    os.makedirs(os.path.join(ckpt_dir, DIR_PROMPTS), exist_ok=True)

    df = pd.read_csv(
        os.path.join(DIR_DATA, input_file_name),
        index_col="task_id",
        keep_default_na=False,
    )

    completions = []
    results = []

    for _ in range(n_participants):
        # Query OpenAI for completions
        for task_id in tqdm(df.index):
                
            prompt, image_path = construct_prompt(df, task_id)
            start_time = time.time()
            completion = query_llm(prompt, image_path)
            end_time = time.time()
            completions.append(completion)
                
            with open(os.path.join(ckpt_dir, DIR_PROMPTS, f"prompt_task_{task_id:03d}.json"), "w") as f:
                prompt_json = {
                    "task_id": task_id,
                    "prompt_text": prompt,
                    "image_path": encode_image(image_path),
                }
                json.dump(prompt_json, f)

            # Parse completions data
            d = {
                "task_id": task_id,
                "runtime": end_time - start_time,
                "rating": parse(completion),
            }
            results.append(d)

    file_name = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_BASELINE, "llm_unaggregated_results.csv")
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # If the file did not exist, write the header first
        if not file_exists:
            writer.writerow(['task_id', 'runtime', 'rating'])

        for row in results:
            
            new_row = [row['task_id'], row['runtime'], row['rating']]

            writer.writerow(new_row)

    return results

def aggregate_responses(experiment_id):
        
        unaggregated_file_name = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_BASELINE, "llm_unaggregated_results.csv")
        df = pd.read_csv(unaggregated_file_name)

        # Function to calculate probabilities
        def calculate_probs(ratings):
            values, counts = np.unique(ratings, return_counts=True)
            probs = counts / counts.sum()
            return probs, values

        # Group by 'task_id' and apply custom operations
        result = df.groupby('task_id').agg({
            'rating': lambda x: calculate_probs(x),
            'runtime': 'mean'
        })

        # Split the rating tuple into two separate columns
        result['probs'], result['support'] = zip(*result['rating'])
        result.drop('rating', axis=1, inplace=True)

        # Reset index to make 'task_id' a column again
        result = result.reset_index()

        def list_to_string(lst):
            """Converts a list to a string with elements separated by commas."""
            return '[' + ', '.join(map(str, lst)) + ']'
        result['probs'] = result['probs'].apply(list_to_string)
        result['support'] = result['support'].apply(list_to_string)

        aggregated_file_name = os.path.join(DIR_EXPERIMENTS, experiment_id, DIR_BASELINE, "llm_aggregated_results.csv")
        result.to_csv(aggregated_file_name, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_name")
    parser.add_argument("experiment_id")
    parser.add_argument("--n_participants", type=int, default=1, help="How many times for each task you want the llm to run")

    args = parser.parse_args()

    run_experiment(
        input_file_name = args.input_file_name,
        experiment_id = args.experiment_id,
        n_participants = args.n_participants,
    )

    aggregate_responses(args.experiment_id)

if __name__ == "__main__":
    main()