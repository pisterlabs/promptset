# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
import openai
import pandas as pd
import time
import argparse

openai.api_key = ""
openai.organization = ""

PROMPT = """
Summarize the following into a medical report having the following sections : "HISTORY OF PRESENT ILLNESS", "PHYSICAL EXAM", "RESULTS", "ASSESSMENT AND PLAN".

{}\n\n\n
"""


def read_data(input_path):
    df = pd.read_csv(input_path)
    return df


# GPT-3 API
def call_gpt(prompt, model='text-davinci-003', stop=None, temperature=0., top_p=1.0,
             max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5

    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            #             ans = openai.ChatCompletion.create
            if model == 'text-davinci-003':
                ans = openai.Completion.create(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions,
                    best_of=requested_completions)
                completions.extend([choice['text'] for choice in ans['choices']])
            elif model == 'gpt-4' or model == 'gpt3.5-turbo':
                ans = openai.ChatCompletion.create(
                    model=model,
                    max_tokens=max_tokens,
                    stop=stop,
                    messages=[
                        {"role": "system",
                         "content": """You are a helpful assistant who summarizes the given input dialogue into a 
                         medical report having the following sections : "HISTORY OF PRESENT ILLNESS", 
                         "PHYSICAL EXAM", "RESULTS", "ASSESSMENT AND PLAN". """},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    n=requested_completions)
                completions.extend([choice['message']['content'] for choice in ans['choices']])
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            time.sleep(min(i ** 2, 60))
    raise RuntimeError('Failed to call GPT API')


def generate_predictions(input_file):
    print('Generating predictions..')
    df = pd.read_csv(input_file)
    output_dict = {
        'TestID': [],
        'SystemOutput': []
    }
    for idx, row in df.iterrows():
        prompt = PROMPT.format(row['dialogue'])
        response = call_gpt(prompt=prompt, model="gpt-4", stop="\n\n\n", temperature=0., top_p=1.0, max_tokens=1024,
                            majority_at=None)
        print('Processed sample : {}'.format(idx))
        output_dict['TestID'].append(row['encounter_id'])
        output_dict['SystemOutput'].append(response[0])
        break
    return output_dict

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    output_dict = generate_predictions(args.input_file)
    df =pd.DataFrame.from_dict(output_dict)
    print('Saving predictions to csv file')
    df.to_csv('taskB_SummQA_run1.csv', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
