import os
import csv
import time 
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from constants import RANKINGS_PROMPT

parser = argparse.ArgumentParser()

parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt-4'], default='gpt-3.5-turbo')
parser.add_argument('--input_csv', type = str, default = 'chatgpt_feedback/without_dolly/test_pairwise_data.csv')
parser.add_argument('--save_feedback_csv', type = str, default = None)
parser.add_argument('--start_index', type = int, default = 0)

args = parser.parse_args()

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\nInput:\n{input}"
    ),
    "prompt_no_input": (
        "{instruction}"
    ),
}

def get_reward(instruction, input, output_1, output_2):
    if str(input) == "":
        print('here')
        instruction = PROMPT_DICT['prompt_no_input'].format(instruction = instruction)
        prompt = RANKINGS_PROMPT.format(instruction = instruction, output_1 = output_1, output_2 = output_2)
    else:
        instruction = PROMPT_DICT['prompt_input'].format(instruction = instruction, input = input)
        prompt = RANKINGS_PROMPT.format(instruction = instruction, output_1 = output_1, output_2 = output_2)

    messages = [{"role": "user", "content": prompt}]

    return messages

def main():

    df = pd.read_csv(args.input_csv)
    df = df.iloc[args.start_index:]

    for j in tqdm(range(len(df))):
        try:
            instruction = df.iloc[j]['instruction']
            input = df.iloc[j]['input']
            output1 = df.iloc[j]['response1']
            output2 = df.iloc[j]['response2']
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = get_reward(instruction, input, output1, output2))
            feedback_1 = completion['choices'][0]['message']['content']
            completion = openai.ChatCompletion.create(
                model = args.gpt_version, 
                messages = get_reward(instruction, input, output2, output1))
            feedback_2 = completion['choices'][0]['message']['content']
            if '(a)' in feedback_1 and '(b)' in feedback_2:
                feedback = '(a)'
            elif '(b)' in feedback_1 and '(a)' in feedback_2:
                feedback = '(b)'
            elif '(a)' in feedback_1 and '(a)' in feedback_2:
                feedback = 'equal'
            elif '(b)' in feedback_1 and '(b)' in feedback_2:
                feedback = 'equal'
            else:
                continue
            print(feedback_1, feedback_2, feedback)
            with open(args.save_feedback_csv, 'a') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerow([instruction, input, output1, output2, feedback])
        except:
            print('Sleeping...')
            time.sleep(5)

if __name__ == '__main__':
    main()
