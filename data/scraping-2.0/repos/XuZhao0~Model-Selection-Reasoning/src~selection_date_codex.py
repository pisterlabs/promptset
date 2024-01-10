import json
import os
import openai
import random
import time
from datetime import datetime
import argparse
import re
from tqdm import tqdm
from typing import Union
from prompts import date_prompt
from collections import OrderedDict, Counter
from tool import *


def get_cot_prompt(data: dict):
    '''
    This function is used to generate the CoT prompt for date understanding problem
    '''
    cot_prompt = date_prompt.CODEX_COT_PROMPT
    inference_prompt = cot_prompt + f'''Q: {data['question'].strip()}

'''
    return inference_prompt


def get_pal_prompt(data: dict):
    '''
    This function is used to generate the PAL prompt for date understanding problem
    '''
    pal_prompt = date_prompt.CODEX_PAL_PROMPT
    inference_prompt = pal_prompt + f'''Q: {data['question'].strip()}

# solution in Python:
'''
    return inference_prompt


def get_select_prompt(data: dict, cot_solution: list, pal_solution: list):
    '''
    This function is used to generate the selection prompt for date understanding problem
    '''
    selection_prompt = date_prompt.CODEX_SELECT_PROMPT

    try:
        pal_generated_list = pal_solution[0].split('"""')
        pal_generated = pal_generated_list[0].strip(
        ) + pal_generated_list[2]
    except Exception as e:
        pal_generated = pal_solution[0]

    cot_generated = cot_solution[0]

    inference_prompt = selection_prompt + f'''Date Understanding Problem: {data['question'].strip()}

Question: Which of the following two choices can correctly answer the math problem?

(A)
{pal_generated.strip()}

(B)
{cot_generated.strip()}

Answer:'''

    return inference_prompt


def query_cot(data: dict, key: str, cot_temperature: float):
    '''
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: OpenAI API key
        cot_temperature: temperature used for CoT

    Returns:
        completions: a list of CoT solutions
    '''
    cot_prompt = get_cot_prompt(data)

    start_time = time.time()
    completions = []
    while True:
        try:

            cot_solution = openai.Completion.create(
                api_key=key,
                model='code-davinci-002',
                max_tokens=500,
                stop='\n\n\n',
                prompt=cot_prompt,
                temperature=cot_temperature,
                top_p=1.0,
                n=1,
                best_of=1)
        except Exception as e:
            cot_solution = None

        if cot_solution is not None:
            completions.extend([choice['text']
                               for choice in cot_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_pal(data: dict, key: str, pal_temperature: float):
    '''
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: OpenAI API key
        pal_temperature: temperature used for PAL

    Returns:
        completions: a list of PAL solutions
    '''
    pal_prompt = get_pal_prompt(data)
    start_time = time.time()
    completions = []
    while True:
        try:
            pal_solution = openai.Completion.create(
                api_key=key,
                model='code-davinci-002',
                max_tokens=500,
                stop='\n\n\n',
                prompt=pal_prompt,
                temperature=pal_temperature,
                top_p=1.0,
                n=1,
                best_of=1)
        except Exception as e:
            pal_solution = None

        if pal_solution is not None:
            completions.extend([choice['text']
                               for choice in pal_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_select(data: dict, key: str, cot_solution: list, pal_solution: list):
    '''
    This function is used to query OpenAI for selection solutions.

    Args:
        data: a dict containing the question and answer
        key: OpenAI API key
        cot_solution: a list of CoT solutions
        pal_solution: a list of PAL solutions

    Returns:
        completions: a list of selection solutions
    '''
    selection_prompt = get_select_prompt(
        data, cot_solution, pal_solution)
    start_time = time.time()
    completions = []
    while True:
        try:
            selection_solution = openai.Completion.create(
                api_key=key,
                model='code-davinci-002',
                max_tokens=100,
                stop='\n\n',
                prompt=selection_prompt,
                temperature=0.,
                top_p=1.0,
                n=1,
                best_of=1)
        except Exception as e:
            selection_solution = None

        if selection_solution is not None:
            completions.extend([choice['text']
                               for choice in selection_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_date(data: dict, key: str, cot_temperature: float, pal_temperature: float):
    '''
    This function is used to query OpenAI for answers in date understanding tasks. It contains three steps:
    1. Query CoT for solutions
    2. Query PAL for solutions
    3. Query model selection answers

    Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.


    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding.

    Returns:
        to_dump_data: a dict containing the question, answer, the final answer and other information
    '''

    cot_ans = None
    pal_ans = None
    selection_ans = None
    final_ans = None
    cot_solution = query_cot(
        data, key, cot_temperature)
    if cot_solution is None:
        print('Time out')
        return None
    else:
        cot_ans = extract_date_cot(cot_solution[0])

    pal_solution = query_pal(
        data, key, pal_temperature)
    if pal_solution is None:
        print('Time out')
        return None
    else:
        pal_ans = execute_date_pal(pal_solution[0])

    if cot_ans is not None and pal_ans is not None:

        # ==== Only select when CoT and PAL are different ====
        if cot_ans != pal_ans:
            selection_ans = query_select(
                data, key, cot_solution=cot_solution, pal_solution=pal_solution)
            if selection_ans is None:
                print('Time out')
                return None
            else:
                selection_choice = extract_choice_codex(selection_ans[0])
                if selection_choice == '(A)':
                    final_ans = pal_ans
                elif selection_choice == '(B)':
                    final_ans = cot_ans
        else:
            final_ans = cot_ans

    elif cot_ans is not None and pal_ans is None:
        final_ans = cot_ans
    elif cot_ans is None and pal_ans is not None:
        final_ans = pal_ans
    else:
        final_ans = None

    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'final_ans': final_ans, 'cot_executed': cot_ans, 'pal_executed': pal_ans,
         'cot_generated': cot_solution, 'pal_generated': pal_solution, 'choice_solution': selection_ans}
    )
    return to_dump_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--cot_temperature', type=float, default=0.)
    parser.add_argument('--pal_temperature', type=float, default=0.)
    parser.add_argument('--output_dir', type=str, default='../output/')
    parser.add_argument(
        '--key', type=str, default='sk-', required=True)

    args = parser.parse_args()

    start_index = args.start
    end_index = args.end
    cot_temperature = args.cot_temperature
    pal_temperature = args.pal_temperature
    output_dir = args.output_dir
    key = args.key

    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")
    dataset = jsonlines_load('../dataset/date_understanding.jsonl')

    # === slice data based on start and end ===
    total_num = len(dataset)
    print('total data: ', total_num)
    if end_index == -1:
        end_index = total_num

    if end_index > total_num:
        end_index = total_num

    tasks = dataset[start_index:end_index]
    task_num = len(tasks)
    print('Current total tasks: ', task_num)

    unfinished_tasks = []

    output_path = os.path.join(output_dir, 'codex/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_path = os.path.join(output_path,
                             f'date_s{start_index}_e{end_index}_{dt_string}.jsonl')

    # === dump data ===
    progress_bar = tqdm(range(task_num))
    for i in range(task_num):
        task = tasks[i]
        start_time = time.time()
        while True:
            try:
                ans = query_date(
                    task, key=key, cot_temperature=cot_temperature,
                    pal_temperature=pal_temperature)
            except Exception as e:
                print(e)
                ans = None

            if ans is not None:
                with open(save_path, "a+") as fout:
                    fout.write(json.dumps(ans)+'\n')
                progress_bar.update(1)
                break
            else:
                sleep_time = random.uniform(3, 5)
                time.sleep(sleep_time)

            if time.time() - start_time > 120:
                print('Time out')
                print('Current Task: ', i)
                unfinished_tasks.append(task)
                break

        sleep_time = random.uniform(3, 5)
        time.sleep(sleep_time)

    end_time_0 = time.time()
    print('Finish at time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f'Time used: {end_time_0 - start_time_0} seconds')

    if len(unfinished_tasks) > 0:
        print('Unfinished tasks: ')
        for task in unfinished_tasks:
            print(task)

    print('Done')
