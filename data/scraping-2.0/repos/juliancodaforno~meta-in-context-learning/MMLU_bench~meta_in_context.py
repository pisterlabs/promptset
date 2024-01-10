import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import openai
import time
from utils import get_category_from_name, get_names_from_category, format_subject, format_example, llm
import argparse
import sys
import os

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num-meta-tasks', type=int, default=2, help='num_meta_tasks')
parser.add_argument('--num-points-per-meta-task', type=int, default=3, help='num_points_per_meta_task')
parser.add_argument('--num-shots-last-task', type=int, default=2, help='num_shots_last_task')
parser.add_argument('--engines', nargs='+', default=['random'], help='List of engines to use which query an llm() function which returns a specific api function later attributed to act()')
args = parser.parse_args()

for engine in args.engines:
    act = llm(engine) #TODO: Change this to your own LLM function
    # create dir 'results/{engine}' if does not exist it;s okay
    os.makedirs(f'results/{engine}', exist_ok=True)
    length = 0
    max_length = 0

    # loop over all tasks
    for task in glob.glob("data/test/*.csv"):
        data = []
        test_df = pd.read_csv(task, header=None)
        task_str = format_subject(task, 5)[1:]

        # test all data points
        for i in range(len(test_df)):
            prompt = ''
            # add meta tasks
            test_category = get_category_from_name(task_str)
            new_list = get_names_from_category(test_category)
            new_list.remove(task_str) # remove actual task
            meta_tasks = np.random.choice(new_list, size=(args.num_meta_tasks,), replace=False)
            for t in range(args.num_meta_tasks):
                prompt += "The following are multiple choice questions (with answers) about {}.\n\n".format(meta_tasks[t])
                path = 'data/dev/' + meta_tasks[t].replace(" ", "_") + '_dev.csv'
                dev_df = pd.read_csv(path, header=None)
                for k in range(args.num_points_per_meta_task):
                    prompt_end = format_example(dev_df, k, include_answer=True)
                    prompt += prompt_end
            # add task
            prompt += "The following are multiple choice questions (with answers) about {}.\n\n".format(task_str)
            dev_df = pd.read_csv(task.replace("test", "dev"), header=None)
            for k in range(args.num_shots_last_task):
                prompt_end = format_example(dev_df, k, include_answer=True)
                prompt += prompt_end

            # add data point
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt += prompt_end
            print(prompt)
            print("===================================")
            length += len(prompt)
            if len(prompt) > max_length:
                max_length = len(prompt)
            choice = act(prompt) # TODO: Change this to your own LLM function

            row = [task, i, choice, test_df.iloc[i, -1], meta_tasks]
            data.append(row)

        df = pd.DataFrame(data, columns=['task', 'item', 'choice', 'gt', 'meta_tasks'])
        df.to_csv(f'results/{engine}/meta_in_context_k=' + str(args.num_shots_last_task) + '_n=' + str(args.num_meta_tasks) + "_t=" + str(args.num_points_per_meta_task) + '_' + task_str + '.csv')

    print(length)
    print(max_length)
