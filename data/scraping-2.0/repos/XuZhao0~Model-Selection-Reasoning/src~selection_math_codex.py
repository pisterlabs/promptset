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
from prompts import math_prompt
from collections import OrderedDict, Counter
from tool import *


def get_cot_prompt(data: dict):
    '''
    This function is used to generate the CoT prompt for math problem
    '''
    cot_prompt = math_prompt.CODEX_COT_PROMPT
    inference_prompt = cot_prompt + f'''Q: {data['question'].strip()}

'''
    return inference_prompt


def get_pal_prompt(data: dict):
    '''
    This function is used to generate the PAL prompt for math problem
    '''
    pal_prompt = math_prompt.CODEX_PAL_PROMPT
    inference_prompt = pal_prompt + f'''Q: {data['question'].strip()}

# solution in Python:
'''
    return inference_prompt


def get_select_prompt(data: dict, cot_solution: list, pal_solution: list):
    '''
    This function is used to generate the selection prompt for math problem
    '''
    selection_prompt = math_prompt.CODEX_SELECT_PROMPT

    try:
        pal_generated_list = pal_solution[0].split('"""')
        pal_generated = pal_generated_list[0].strip(
        ) + pal_generated_list[2]
    except Exception as e:
        pal_generated = pal_solution[0]

    cot_generated = cot_solution[0]

    inference_prompt = selection_prompt + f'''Math Problem: {data['question'].strip()}

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


def query_math(data: dict, key: str, cot_temperature: float, pal_temperature: float, sc_num: int):
    '''
    This function is used to query OpenAI for answers in arithmetic tasks. It contains three steps:
    1. Query CoT for solutions
    2. Query PAL for solutions
    3. Query model selection answers

    Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.

    We also use majority voting to select the final answer if we have multiple self-consistency samples.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT. 0 for greedy decoding. We set it to 0.5 for self-consistency samples.
        pal_temperature: the temperature used in PAL. 0 for greedy decoding. We set it to 0.8 for self-consistency samples.
        sc_num: the number of self-consistency samples

    Returns:
        to_dump_data: a dict containing the question, answer, the final answer and other information
    '''

    cot_answers = []
    pal_answers = []
    cot_solutions = []
    pal_solutions = []
    selection_solutions = []
    final_answers = []

    for i in range(sc_num):
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
            cot_ans = extract_num_codex(cot_solution[0])
            cot_answers.append(cot_ans)
            cot_solutions.append(cot_solution[0])

        pal_solution = query_pal(
            data, key, pal_temperature)
        if pal_solution is None:
            print('Time out')
            return None
        else:
            pal_ans = safe_execute_codex(pal_solution[0])
            pal_answers.append(pal_ans)
            pal_solutions.append(pal_solution[0])

        if cot_ans is not None and pal_ans is not None:

            # ==== Only select when CoT and PAL are different ====
            if abs(cot_ans - pal_ans) > 1e-3:
                selection_ans = query_select(
                    data, key, cot_solution=cot_solution, pal_solution=pal_solution)
                if selection_ans is None:
                    print('Time out')
                    return None
                else:
                    selection_choice = extract_choice_codex(selection_ans[0])
                    selection_solutions.append(selection_ans[0])
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

        final_answers.append(final_ans)

    count = Counter(final_answers)
    majority_ans = count.most_common(1)[0][0]

    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'majority_ans': majority_ans, 'final_answers': final_answers,
         'cot_executed': cot_answers, 'pal_executed': pal_answers,
         'cot_generated': cot_solutions, 'pal_generated': pal_solutions, 'choice_solution': selection_solutions}
    )

    return to_dump_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--dataset', type=str, choices=[
                        'gsm8k', 'svamp', 'asdiv', 'singleeq', 'singleop',
                        'singleaddsub', 'multiarith'], default='gsm8k')
    parser.add_argument('--cot_temperature', type=float, default=0.)
    parser.add_argument('--pal_temperature', type=float, default=0.)
    parser.add_argument('--sc_num', type=int, default=1,
                        help='Self-consistency samples. 1 indicates greedy decoding')
    parser.add_argument('--output_dir', type=str, default='../output/')
    parser.add_argument(
        '--key', type=str, default='sk-', required=True)

    args = parser.parse_args()

    start_index = args.start
    end_index = args.end
    dataset_name = args.dataset
    cot_temperature = args.cot_temperature
    pal_temperature = args.pal_temperature
    sc_num = args.sc_num
    output_dir = args.output_dir
    key = args.key

    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")

    if dataset_name == 'gsm8k':
        dataset = jsonlines_load('../dataset/gsm8K_test.jsonl')
    elif dataset_name == 'svamp':
        dataset = jsonlines_load('../dataset/svamp.jsonl')
    elif dataset_name == 'asdiv':
        dataset = jsonlines_load('../dataset/asdiv.jsonl')
    elif dataset_name == 'singleeq':
        dataset = jsonlines_load('../dataset/single_eq.jsonl')
    elif dataset_name == 'singleop':
        dataset = jsonlines_load('../dataset/single_op.jsonl')
    elif dataset_name == 'singleaddsub':
        dataset = jsonlines_load('../dataset/single_addsub.jsonl')
    elif dataset_name == 'multiarith':
        dataset = jsonlines_load('../dataset/multiarith.jsonl')

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
                             f'{dataset_name}_sc{sc_num}_s{start_index}_e{end_index}_{dt_string}.jsonl')

    # === dump data ===
    progress_bar = tqdm(range(task_num))
    for i in range(task_num):
        task = tasks[i]
        wait_time = min(sc_num * 100, 360)
        start_time = time.time()
        while True:
            try:
                ans = query_math(
                    task, key=key, cot_temperature=cot_temperature,
                    pal_temperature=pal_temperature, sc_num=sc_num)
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

            if time.time() - start_time > wait_time:
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
