import os
import csv
import time 
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from constants import RATINGS_PROMPT

parser = argparse.ArgumentParser()

parser.add_argument('--start_index', type = int, default = 0)
parser.add_argument('--input_csv', type = str, default = 'ratings_sample_generation.json')
parser.add_argument('--save_feedback_csv', type = str, default = None)
parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt-4'], default='gpt-3.5-turbo')

args = parser.parse_args()

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\nInput:\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}

def get_reward(instruction, input, output):
    if str(input) == "":
        instruction = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
        prompt = RATINGS_PROMPT.format(instruction = instruction, response = output)
    else:
        instruction = PROMPT_DICT['prompt_input'].format(instruction = instruction, input = input)
        prompt = RATINGS_PROMPT.format(instruction = instruction, response = output)
    
    messages = [{"role": "user", "content": prompt}]

    return messages
        
def main():

    df = pd.read_csv(args.input_csv)
    df = df.iloc[args.start_index:]
    total = 0
    for j in tqdm(range(len(df))):
        instruction = df.iloc[j]['instruction']
        input  = df.iloc[j]['input']
        output = df.iloc[j]['response']
        try:
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = get_reward(instruction, input, output)
            )
            feedback = completion['choices'][0]['message']['content']
            score  = feedback.split("\n")[0]
            if score.isnumeric():
                score = int(score)
                print(score)
                with open(args.save_feedback_csv, 'a') as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([instruction, input, output, score])
        except:
            ### in case the API goes into error, skip the instance instead of exponential backoff as repeated requests is not cost-efficient.
            print('Sleeping...')
            time.sleep(5)

if __name__ == '__main__':
    main()
