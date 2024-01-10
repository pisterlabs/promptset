import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import get_category_from_name, get_names_from_category, llm, format_subject, format_example
import argparse
import openai
import os

#Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_points_per_task', type=int, default=[6,9], help='num_points_per_task')
parser.add_argument('--engines', nargs='+', default=['random'], help='List of engines to use which query an llm() function which returns a specific api function later attributed to act()')
args = parser.parse_args()

# loop over all engines names which query different LLM functions
for engine in args.engines:
    act = llm(engine) #TODO: Change this to your own LLM function
    # create dir 'results/{engine}' if does not exist it;s okay
    os.makedirs(f'results/{engine}', exist_ok=True)
    length = 0
    max_length = 0
    tasks = get_names_from_category('STEM')

    # loop over all tasks
    for task_x in tasks:
        task = "data/test/" + task_x.replace(" ", "_") + '_test.csv'
        data = []
        test_df = pd.read_csv(task, header=None)
        task_str = format_subject(task, 5)[1:]

        # test all data points
        start_idx = 0
        #if results/{engine}/matched_in_context_stem_{task_str}.csv exists, then start from the next index of the last row
        if os.path.exists(f'results/{engine}/matched_in_context_stem_' + task_str + '.csv'):
            df = pd.read_csv(f'results/{engine}/matched_in_context_stem_' + task_str + '.csv')
            start_idx = df.iloc[-1]['item'] + 1
            print(f"starting from index {start_idx}")
            data = df.values.tolist()
        for i in range(start_idx, len(test_df)):
            prompt = ''
            
            # add dev tasks
            prompt += "The following are multiple choice questions (with answers) about {}.\n\n".format(task_str)
            dev_df = pd.read_csv(task.replace("test", "dev"), header=None) 
            for k in range(5): # Because only 5 dev points so add one from val as well
                prompt_end = format_example(dev_df, k, include_answer=True)
                prompt += prompt_end

            # add the one  val task
            val_df = pd.read_csv(task.replace("test", "val"), header=None)
            prompt_end = format_example(val_df, 0, include_answer=True)
            prompt += prompt_end

            # add data point
            test_prompt = format_example(test_df, i, include_answer=False)
            for k in range(args.num_points_per_task[0], args.num_points_per_task[1]):
                #Randomly sample an index which can't be i and is less than len(test_df)
                rdm_test_idx = np.random.choice(np.arange(len(test_df)))
                while rdm_test_idx == i:
                    rdm_test_idx = np.random.choice(np.arange(len(test_df)))

                prompt_end = format_example(test_df, rdm_test_idx, include_answer=True)
                prompt += prompt_end
                choice = act(prompt + test_prompt) 
                row = [task, i, choice, test_df.iloc[i, -1], k]
                data.append(row)

        df = pd.DataFrame(data, columns=['task', 'item', 'choice', 'gt', 'num_previous_points'])
        df.to_csv(f'results/{engine}/matched_in_context_stem_' + task_str + '.csv')

