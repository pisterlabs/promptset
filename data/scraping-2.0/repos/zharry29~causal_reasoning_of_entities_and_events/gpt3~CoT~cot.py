import os 
import json
import time
import openai
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

openai.api_key_path = '../../api_keys/harry.key'

def gpt3(prompt):
    ret = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        logprobs=5,
        stop='\n',
    )

    gen_text = ret["choices"][0]["text"] 
    return gen_text


def label2idx(label):
    if 'less' in label:
        return 0
    elif 'equally' in label:
        return 1
    elif 'more' in label:
        return 2
    else:
        return -1


parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_shots', type=int, default=3, help='number of shots for GPT-3'
)
parser.add_argument(
    '--cot', type=int, help='choose to use cot (1) or not (0)'
)
parser.add_argument(
    '--seed', type=int, help='set seed for sampling prompts'
)
parser.add_argument(
    '--dataset', type=str, help='choose from "dev" and "test"'
)

def main():
    args = parser.parse_args()
    np.random.seed(24)

    with open(f'../../data_{args.dataset}_v2.json', 'r') as f:
        dev_data = json.load(f)
    f.close()

    if args.cot:
        all_prompts = open('./cot-prompts.txt', 'r').read().split('\n\n\n')
    else:
        all_prompts = open('./prompts.txt', 'r').read().split('\n\n\n')

    cur_prompts_idx = np.random.choice(len(all_prompts), args.n_shots, replace=False)
    prompt = '\n\n\n'.join([all_prompts[idx] for idx in cur_prompts_idx])

    gt_answers_all = []
    pred_answers_all = []
    for proc_id, proc in tqdm(dev_data.items(), position=0, leave=False):
        goal = proc['goal']
        if goal[-1] == '.':
            goal = goal[:-1]

        step_lst = proc['steps']
        questions = {}
        steps = []
        for idx, step_sublst in enumerate(step_lst):
            for step_dict in step_sublst:
                if idx != 0:
                    if step_dict.get('type') == 'step':
                        steps.append(step_dict['step'].strip())
                if step_dict.get('type') == 'event':
                    if (cur_event := step_dict['event']) not in questions:
                        if cur_event.strip()[-1] == '.':
                            cur_event = cur_event[:-1]
                        questions[cur_event] = f"What is the likelihood that {cur_event.lower().strip()}?"
        
        gt_answers = []
        pred_answers = []
        cur_goal = f"Goal: {goal.lower()}."
        for i, (event, event_q) in enumerate(tqdm(questions.items(), position=1, leave=False)):
            cur_ans_gt = [1 for _ in range(len(step_lst)-1)]
            cur_pred_ans = [1 for _ in range(len(step_lst)-1)]
            question = f"Question: {event_q}"
            cur_prompt = prompt + '\n\n\n'

            for step_idx, step_sublst in enumerate(tqdm(step_lst[1:], position=2, leave=False)):

                context = f"Context: {' '.join(steps[:step_idx+1])}"
                cur_prompt += cur_goal + '\n' + context + '\n' + question + '\nAnswer:'
                pred = gpt3(cur_prompt)
                cur_prompt += pred + '\n\n'
                if args.cot:
                    pred = pred.split('the likelihood change is')[-1].strip().replace('"', '').replace('.', '')
                cur_pred_ans[step_idx] = label2idx(pred)

                for step_dict in step_sublst:
                    if (cur_event := step_dict.get('event')):
                        if cur_event.strip()[-1] == '.':
                            cur_event = cur_event[:-1]
                        if cur_event == event:
                            cur_ans_gt[step_idx] = label2idx(step_dict['change'].lower())
            gt_answers.append(cur_ans_gt)
            pred_answers.append(cur_pred_ans)

        gt_answers_all += gt_answers
        pred_answers_all += pred_answers

    with open(f'./results/{args.dataset}/cot_{args.cot}_{args.n_shots}_gt_seed_{args.seed}.pkl', 'wb') as f:
        pickle.dump(gt_answers_all, f)
    f.close()
    
    with open(f'./results/{args.dataset}/cot_{args.cot}_{args.n_shots}_pred_seed_{args.seed}.pkl', 'wb') as f:
        pickle.dump(pred_answers_all, f)
    f.close()


if __name__ == '__main__':
    main()