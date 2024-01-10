#!/usr/bin/env python

import os
import time
import argparse
import html2text

import openai
import pandas as pd

from src.execution import check_correctness
from src.utils.files import json2data
from tqdm import tqdm 


def save_generation_results(dataframe, args):
    print(f'ensure that {args.output_dir} directory exists...')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'successfully ensured directory existence')

    config_version = args.config.split("config/")[1].split(".json")[0]
    filename = f"{args.model}_{config_version}_result_new.csv"
    result_filename = os.path.join(args.output_dir, filename)
    print("Saving results to ", result_filename)
    dataframe.to_csv(result_filename)


def getConcepts(problem_id, df):
    """ 
        return a list of concepts that required for solving this problem 
    """
    concept_list = ['input_str', 'input_cast', 'output', 'assignment', 'conditional',
       'function_call', 'function_def', 'function_return', 'loop_counting',
       'loop_until', 'loop_elements', 'loop_nested', 'stat_calculate',
       'file_read', 'file_write', 'list', 'list_2d', 'dictionary', 'item_set',
       'tuple']
    return [concept for concept in concept_list if (df[df['id']==problem_id][concept].iloc[0]==1)]


def generatePropmt(problem_description, skeleton_code):
    
    # prompt = "You are a helpful Teaching Assistant in a CS1 programming course teaching the basics of python programming."
    prompt = "Bellow is a problem statement,"
    prompt += f" write a program in Python that solves the problem."
    prompt += "Put your code solution within fenced code blocks,"
    prompt += " and do not provide explanations for your solution. \n"
    prompt += f"{problem_description}\n{skeleton_code}"

    return prompt

def extractResultCode(raw_result):
    """
    cleaning the generated code format
    """
    if raw_result[:9] == "```python":
        raw_result = raw_result.split("```python")[1]
    if raw_result[-3:] == "```":
        raw_result = raw_result.split("```")[0]
    return raw_result

def generateGPTAnswer(prompt, api_key=os.environ.get('OPEN_AI_KEY', None), model="gpt-3.5-turbo"):
    """
    input: prompt, problem_description, student_code
    output: gpt-genrated code
    """
    openai.api_key = api_key
    
    gpt_code  = ""
    
    if model in ["code-davinci-edit-001"]: # endpoint is v1/completions
        response = openai.Edit.create(
            model=model,
            input="",
            temperature=0,
            instruction=prompt
        )
        
        gpt_code = response.choices[0].text.strip()
        print(gpt_code)

    
    else: # endpointpython src/codeGenerating.py -i /scratch/work/koutchc1/datasets/falconcode -o /home/koutchc1/learnlab2023/outputs/ -m gpt-4 is v1/chat/completions
        response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", 
            "content": prompt
            }
        ]
        )
        gpt_code = response["choices"][0]['message']['content']
        
    return gpt_code


def get_results(problems_df, model):
    print("problems dataframe", problems_df)

    results = []
    for i in tqdm(range(len(problems_df))):
        row = problems_df.iloc[i].to_dict()
        problem_description = html2text.html2text(row['prompt'])
        row["prompt"] = generatePropmt(problem_description, row['skeleton'])
        row["model"] = model
        try: 
            gpt_answser = generateGPTAnswer(row["prompt"], model=model)
            row["code"] = extractResultCode(gpt_answser)
            output = check_correctness(row, timeout=5.0, completion=None)
            row.update(output)
            
        except openai.error.ServiceUnavailableError:
            print("Service unavailable, trying again")
            continue
        except openai.error.RateLimitError:
            print("No moneey")
            time.sleep(1)
            exit()

        results.append(row)
        
    result_df = pd.DataFrame(results)

    return result_df



def query_openai(problems_df, args):
    dataframe = []
    remaining, n_trials = len(problems_df),  3
    while remaining > 0 and n_trials > 0:
        result_df = get_results(problems_df, args.model)
        remaining -= len(result_df)
        dataframe.append(result_df)
        problems_df = problems_df[~problems_df['id'].isin(result_df["id"])]
        print("Remaining", remaining, "n_trials", n_trials)
        print("sleeping before trying again")
        time.sleep(60)

    dataframe = pd.concat(dataframe, axis=0, ignore_index=True)
    return dataframe




def parse_args():
    parser = argparse.ArgumentParser(description='Call chatGPT-3.5 api to generate code and save into a csv file.')
    parser.add_argument('-m', '--model', help='the model used to generate the code', default='gpt-3.5-turbo')
    parser.add_argument('-i', '--input-dir', required=True, help='the directory to save result.csv file')
    parser.add_argument('-o', '--output-dir',  required=True, help='the directory to save result.csv file')
    parser.add_argument('-c', '--config', help='the configuration', default='config/v1.json')
    parser.add_argument('-v', '--validity', help='the valid ones', default='config/validity.json')
    
    return parser.parse_args()

def main():
    args = parse_args()
    problems_df = load_dataset(args)
    # Temporary
    problems_df = problems_df.head(1)
    dataframe = query_openai(problems_df, args)
    # save_generation_results(dataframe, args)
    

if __name__ == '__main__':
    main()
