import os
import openai
import argparse
from tqdm import tqdm
import pickle
import sys
sys.path.append('../')

#from GPT_prompts import prompts
from asp_domains import babi_tasks
from asp_modules import *
from asp_modules import asp_modules_dict

from keys import API_KEY,ORG_KEY

from task_config import tasks_dict, data_paths, Config
from babi_utils import get_query, GPT_answer, ASP_solve, process_answer_set, write_debug_COT, str_to_int

parser = argparse.ArgumentParser()

parser.add_argument('--n', default = 100, type = int, help = 'Specifies the number of examples to run.')
parser.add_argument('--task', default = 1, type = int, help = 'Specifies task number (1-20).')
parser.add_argument('--ALL', action = 'store_true', help = 'Specifies all 20 tasks.')
parser.add_argument('--model', default = 'text-davinci-003', type = str, help = 'Specifies model (supports "text-davinci-002" and "text-davinci-003")')


args=parser.parse_args()


gpt_model = args.model 
prompt_version = 3

# Assign openai API key
openai.organization = ORG_KEY
openai.api_key = API_KEY

#breakpoint()
# Load previous GPT-3 responses
if f'prompt_cache_CoT_{gpt_model}_v{prompt_version}.pickle' in os.listdir():
    with open(f'prompt_cache_CoT_{gpt_model}_v{prompt_version}.pickle', 'rb') as handle:
        prompt_cache = pickle.load(handle)
else:
    prompt_cache = dict()


task_nums = [args.task]
if args.ALL:
    task_nums = [i+1 for i in range(0,20)]
else:
    task_nums = [args.task]

predictions=[];answers=[]
accuracies=list()
for task_idx in task_nums:
    
    # Read Data
    data_path = os.path.join('../bAbI/data',data_paths[task_idx])
    with open(data_path) as f:
        lines = f.readlines()
    beginning_lines=[idx for idx,line in enumerate(lines) if int(line[:2])==1]
    question_lines = [idx for idx,line in enumerate(lines) if '?' in line]
    
    with open(f'COT_prompts_v{prompt_version}/{task_idx}.txt') as f:
        instructions = f.read()
    
    # Load task configuration object
    config=Config(tasks_dict[task_idx])
    
    print(f"Evaluating task {task_idx}")
    task_rules = babi_tasks[task_idx]
    # Load task-relevant modules 
    module_rules = '\n\n'.join([asp_modules_dict[module] for module in config.modules])
    print(f"Using modules {config.modules}")

    # Main Evaluation Loop
    stop = config.stop
    parse_word = config.parse_word
    correct,total=0,0
    wrong_indices=list()
    pbar= tqdm(range(0, args.n))
    debug=list()
    for eval_num in pbar:
        # 1. Process data instance and prepare data for GPT-3
        context, query, answer  = get_query(eval_num,lines, beginning_lines , question_lines)
        # 2. Call GPT-3
        pred_full, prompt_cache = GPT_answer(instructions, context, query, gpt_model, config, prompt_cache)
        if 'answer is' in pred_full:
            pred = pred_full[pred_full.index('answer is')+9:].strip().replace('.','').lower()
        else:
            pred=pred_full
        if pred in str_to_int:
            pred = str(str_to_int[pred])
        if pred.lower()==answer.lower():
            correct+=1
            debug.append([eval_num, context, query, None, None, pred_full, pred, answer, '(correct)'])
        else:
            wrong_indices.append(eval_num)
            debug.append([eval_num, context, query, None, None, pred_full, pred, answer, '(wrong)'])
        
        predictions.append(pred)
        answers.append(answer)
        total+=1
    
        pbar.set_description(f'acc: {100*correct / total}')
        
    #write_debug_COT(debug, correct, total, task_idx, 'COT', f'{gpt_model}_v{prompt_version}_')
    # Save GPT-3 responses
    accuracies.append(100*correct/total)
    with open(f'prompt_cache_CoT_{gpt_model}_v{prompt_version}.pickle', 'wb') as handle:
        pickle.dump(prompt_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
