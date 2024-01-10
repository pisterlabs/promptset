import re
import ast
import json
import time
import utils
import pickle
import random
import openai
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import accuracy_score


class OpenPI():
    def __init__(self, metadata, apply_template):
        self.metadata = metadata
        self.apply_template = apply_template
    
    def build_text_prompt(self, max_len):
        train_dict = self.metadata['train'] 
        prompt = ''
        cur_len = 0
        counter = 0 

        while cur_len + max_len <= args.context_size:
            cur_idx = str(np.random.choice(len(train_dict), 1, replace=False)[0])
            event_dict = train_dict[cur_idx]
            goal = event_dict['goal']
            steps = event_dict['steps']
            cur_prompt = apply_template(goal, steps) 
            cur_len += utils.gpt3_tokenizer(cur_prompt)
            prompt += cur_prompt
            counter += 1

        print(f'Total samples in prompt: {counter}')
        print(f'Average tokens per sample: {cur_len / counter}')
        return prompt

    def build_code_prompt(self, max_len):
        train_dict = self.metadata['train'] 
        prompt = ''
        cur_len = 0
        for event_dict in train_dict.values():
            goal = event_dict['goal']
            steps = event_dict['steps']
            cur_prompt = apply_template(goal, steps) 
            cur_prompt = [' '.join(lst) for lst in cur_prompt]
            for sub_prompt in cur_prompt:
                cur_len += utils.gpt3_tokenizer(sub_prompt)
                if cur_len + max_len > args.context_size:
                    break
                else:
                    prompt += sub_prompt + '\n\n'
        return prompt
    
    def run_llm(self, prompt, model, temperature=0.7, stop=['\n\n']):
        model_name = {
            "davinci": "text-davinci-002",
            "davinci003": "text-davinci-003",
            "davinci-old": "davinci",
            "curie": "text-curie-001",
            "codex": "code-davinci-002",
            "ada": "text-ada-001"
        }
        while True:
            try:
                ret = openai.Completion.create(
                    engine=model_name[model],
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=300,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=stop
                )
                break
            except Exception as e:
                print(e)
                print("Retrying in 10 seconds")
                time.sleep(10)

        gen_text = ret["choices"][0]["text"].strip()#.split('\n')[0]
        return gen_text
    
    def predict(self):
        pred_name, gold_name = get_fname() 
        val_data = self.metadata['test']
        max_len = compute_longest_prompt(val_data, self.apply_template) 
        print(max_len)

        if args.prompt == "text":
            prompt = self.build_text_prompt(max_len)
        elif args.prompt == "code":
            if args.style == 'comment':
                prompt = open(f'./code-prompts/comment_prefix.py').read()
            elif args.style == 'class':
                prompt = open(f'./code-prompts/class_prefix.py').read()
            else:
                prompt = ''
            prompt += self.build_code_prompt(max_len)
        
        preds = []
        golds = []
        for id, example in tqdm(val_data.items(), position=0, leave=False):
            goal = example['goal']
            steps = example['steps']
            if args.prompt == 'text':
                cur_prompt = prompt + f'Goal: {goal}\n\n'
                for i, step in enumerate(tqdm(steps, position=1, leave=False)):
                    step_state_changes = step['state_changes']
                    step_desc = step['description']
                    cur_prompt += f'Step {i+1}: {step_desc}\n\n'
                    cur_prompt += f'Entity status changes:\n'
                    llm_pred = self.run_llm(cur_prompt, args.model, stop='\n\n')
                    llm_pred = utils.parse_preds(llm_pred)
                    if len(llm_pred) == 0:
                        llm_pred.append('There will be no change.')
                    for line in llm_pred:
                        # entries = re.match(r'The (.*) is (.*) before and (.*) afterwards.', line)
                        # print((entries.group(1), entries.group(2), entries.group(3)))
                        cur_prompt += '- ' + line.strip() + '\n'
                    cur_prompt += '\n'
                    preds.append({'id': f'{str(id)}||{str(i+1)}', 'answers': llm_pred})

                    state_change_lst = []
                    if step_state_changes:
                        for state_change in step_state_changes:
                            entity, attr, pre, post = state_change
                            state_change_lst.append(f'The {attr.split("|")[0].strip()}) of ({entity.split("|")[0].strip()}) is ({pre.split("|")[0].strip()}) before and ({post.split("|")[0].strip()}) afterwards.')
                    golds.append({'id': f'{str(id)}||{str(i+1)}', 'answers': state_change_lst})

            elif args.prompt == 'code':
                cur_template = apply_template(goal, steps)
                prompt_template = [lst[0] for lst in cur_template]
                cur_gold = [ast.literal_eval(lst[1].strip()) for lst in cur_template]
                for i, step_prompt in enumerate(tqdm(prompt_template, position=1, leave=False)):
                    cur_prompt = prompt + step_prompt
                    llm_pred = self.run_llm(cur_prompt, args.model, stop='\n\n').strip()
                    print(llm_pred[-1])

                    if llm_pred[-1] in ["'", '"']:
                        llm_pred += ']'
                    
                    try:
                        llm_pred = ast.literal_eval(llm_pred)
                    except:
                        llm_pred = ','.join(llm_pred.split(',')[:-1]) + ']'
                        try:
                            llm_pred = ast.literal_eval(llm_pred)
                        except:
                            llm_pred = []
                            
                    if len(llm_pred) == 0:
                        llm_pred.append('There will be no change.')
                    preds.append({'id': f'{str(id)}||{str(i+1)}', 'answers': llm_pred})
                golds += [{'id': f'{str(id)}||{str(i+1)}', 'answers': g} for i, g in enumerate(cur_gold)]

        with open(f'./result/test_{pred_name}.jsonl', 'w') as f:
            for d in preds:
                json.dump(d, f)
                f.write('\n')
        with open(f'./result/test_{gold_name}.jsonl', 'w') as f:
            for d in golds:
                json.dump(d, f)
                f.write('\n')


def apply_text_template(goal, steps, train=True):
    prompt = f'Goal: {goal}\n\n'
    step_prompts = []

    for i, step in enumerate(steps):
        cur_prompt = ''
        step_state_changes = step['state_changes']
        step_desc = step['description']
        cur_prompt += f'Step {i+1}: {step_desc}\n\n'
        cur_prompt += f'Entity status changes:\n'
        if train:
            if step_state_changes:
                for state_change in step_state_changes:
                    entity, attr, pre, post = state_change
                    if '|' in entity:
                        entity = utils.choose_openpi_options(entity)
                    if '|' in attr:
                        attr = utils.choose_openpi_options(attr)
                    if '|' in pre:
                        pre = utils.choose_openpi_options(pre)
                    if '|' in post:
                        post = utils.choose_openpi_options(post)
                    cur_prompt += f'- The {attr} of {entity} is {pre} before and {post} afterwards.\n'
                # cur_prompt += '\n'
            else:
                cur_prompt += '- There will be no change.\n'
        step_prompts.append(cur_prompt)
    
    if train:
        prompt += '\n'.join(step_prompts)
    else:
        prompt = step_prompts
    return prompt + '\n'


def apply_code_template(goal, steps):

    with open(f'./code-prompts/{args.style}' + '.py') as f:
        template = f.read()
    f.close()

    contexts = ['start.']
    cur_steps = []
    cur_states = []
    for i, step in enumerate(steps):
        step_state_changes = step['state_changes']
        if step_state_changes:
                for state_change in step_state_changes:
                    entity, attr, pre, post = state_change
                    if '|' in entity:
                        entity = utils.choose_openpi_options(entity)
                    if '|' in attr:
                        attr = utils.choose_openpi_options(attr)
                    if '|' in pre:
                        pre = utils.choose_openpi_options(pre)
                    if '|' in post:
                        post = utils.choose_openpi_options(post)
                    cur_states.append(f'the {attr} of {entity} is {pre} before and {post} afterwards.')
                # cur_prompt += '\n'
        else:
            cur_states.append('there will be no change.\n')
        cur_steps.append(step['description'])
        contexts.append(contexts[-1] + ' ' + cur_steps[-1].lower())

    all_prompts = []
    for context, cur_step in zip(contexts, cur_steps):
        ret = []
        for t in template.split('$'):
            ret.append(t.replace("{goal}", goal).replace("{context}", context).replace("{current_step}", cur_step).replace("{answer}", str(cur_states)))
        all_prompts.append(ret)
    return all_prompts


def compute_longest_prompt(val_data, apply_template):
    max_len = 0
    for key, val in tqdm(val_data.items()):
        goal = val['goal']
        steps = val['steps']
        cur_prompt = apply_template(goal, steps)
        if args.prompt == 'code':
            cur_prompt = [' '.join(lst) for lst in cur_prompt]
            for sub_prompt in cur_prompt:
                cur_len = utils.gpt3_tokenizer(sub_prompt + '\n\nAnswer:')
                if cur_len > max_len:
                    max_len = cur_len
        else:
            cur_len = utils.gpt3_tokenizer(cur_prompt + '\n\nAnswer:')
            if cur_len > max_len:
                max_len = cur_len
    return max_len

def get_fname():
    if not args.style:
        pred_name = f'{args.prompt}_{args.model}_pred_{args.context_size}_{args.seed}'
    else:
        pred_name = f'{args.prompt}_{args.style}_{args.model}_pred_{args.context_size}'
    
    if not args.style:
        gold_name = f'{args.prompt}_{args.model}_gold_{args.context_size}_{args.seed}'
    else:
        gold_name = f'{args.prompt}_{args.style}_{args.model}_gold_{args.context_size}'
    
    return pred_name, gold_name


parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, help='Either text or code.')
parser.add_argument('--model', type=str, help='Either davinci, curie or codex.')
parser.add_argument('--data_path', type=str, help='Path to the folder that stores parsed OpenPI datasets.')
parser.add_argument('--style', type=str, help='choose style of code prompt from one of ["vanilla", "good_var_name", "with_comments", "class_obj"]')
parser.add_argument('--context_size', type=int, help='token threshold for GPT3 context prompt.')
parser.add_argument('--completion_size', type=int, help='token threshold for GPT3 completion.')
parser.add_argument('--seed', type=int, default=None, help='random seed')
#parser.add_argument('--dataset', type=str, help='Name of the datasset')
#parser.add_argument('--xxx', action='store_true', help='')
parser.add_argument('--key', type=str, help='The name of the OpenAI API key file.')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)

    openai.api_key_path = f'../../_private/{args.key}.key'

    data_name = 'anli'
    NUM_EXAMPLES_IN_PROMPT = 2

    meta_data = utils.load_meta_data(args.data_path)

    if args.prompt == "text":
        apply_template = apply_text_template
    elif args.prompt == "code":
        apply_template = apply_code_template

    inference_model = OpenPI(meta_data, apply_template)
    inference_model.predict()










